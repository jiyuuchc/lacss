#!/usr/bin/env python3

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import json
from os.path import join

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

import lacss

try:
    from . import data
except:
    import data

import typer

app = typer.Typer(pretty_exceptions_enable=False)


def format_array(arr):
    s = [f"{v:.4f}" for v in arr]
    s = ", ".join(s)
    return s


def mask_ious(pred, gt_mask_indices, box_ious):

    n_pred = box_ious.shape[0]
    patches = np.pad(pred["instance_output"] >= 0.5, [[0, 0], [1, 1], [1, 1]])
    yc = np.asarray(pred["instance_yc"])
    xc = np.asarray(pred["instance_xc"])

    # gt_areas = np.count_nonzero(gt_instances, axis=(1, 2))
    _, indices, gt_areas = np.unique(
        gt_mask_indices[:, 0], return_index=True, return_counts=True
    )
    indices = indices.tolist() + [len(gt_mask_indices[:, 0])]
    pred_areas = np.count_nonzero(patches[:n_pred], axis=(1, 2))
    total_areas = pred_areas[:, None] + gt_areas

    its = np.zeros_like(box_ious)
    for pred_idx, gt_idx in np.stack(np.where(box_ious > 0), axis=-1):
        gt_indices = gt_mask_indices[indices[gt_idx] : indices[gt_idx + 1], 1:] - (
            yc[pred_idx, 0, 0],
            xc[pred_idx, 0, 0],
        )
        gt_indices = np.clip(gt_indices + 1, 0, yc.shape[-1] + 1)
        gt_indices = np.swapaxes(gt_indices, 0, 1)
        pred = patches[pred_idx]

        its[pred_idx, gt_idx] = np.count_nonzero(pred[tuple(gt_indices)])

    ious = its / (total_areas - its + 1e-6)

    return ious


@app.command()
def run_test(
    checkpoint: str,
    datapath: str = "../../livecell_dataset",
    logpath: str = "./",
    supervised: bool = False,
    coco: bool = False,
):

    print("evaluating...")

    try:
        trainer = lacss.train.Trainer.from_checkpoint(checkpoint)
    except:
        from lacss.deploy import load_from_pretrained

        module, params = load_from_pretrained(checkpoint)
        trainer = lacss.train.Trainer(module.bind(dict(params=params)))

    print(f"checkpoint loaded from {checkpoint}")

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    mask_ap = lacss.metrics.AP(thresholds)
    box_ap = lacss.metrics.AP(thresholds)
    fg_acc_sum = 0
    cnts = 0

    for c in range(8):

        mask_ap_c = lacss.metrics.AP(thresholds)
        box_ap_c = lacss.metrics.AP(thresholds)
        fg_acc_sum_c = 0
        cnts_c = 0

        test_data = data.test_data(
            datapath,
            supervised=supervised,
            coco=coco,
            cell_type=c,
        )

        if not trainer.initialized:
            trainer.initialize(test_data, tx=optax.set_to_zero())

        print(f" Cell type : {c}")

        for inputs, labels in tqdm(test_data()):

            pred = trainer(inputs, strategy=lacss.train.strategy.Core)
            scores = np.asarray(pred["pred_scores"])
            valid_preds = scores >= 0
            scores = scores[valid_preds]

            # boxes
            pred_boxes = np.asarray(lacss.ops.bboxes_of_patches(pred))
            pred_boxes = pred_boxes[valid_preds]
            gt_boxes = np.asarray(labels["gt_boxes"])
            box_ious = np.asarray(lacss.ops.box_iou_similarity(pred_boxes, gt_boxes))
            box_ap_c.update(box_ious, scores)
            box_ap.update(box_ious, scores)

            # masks
            gt_mask_indices = np.asarray(labels["gt_mask_indices"])
            m_ious = mask_ious(pred, gt_mask_indices, box_ious)
            mask_ap_c.update(m_ious, scores)
            mask_ap.update(m_ious, scores)

            if not supervised:
                # fg_acc
                pred_masks = pred["fg_pred"] >= 0
                gt_masks = labels["gt_mask"]
                fg_acc_sum_c += (pred_masks == gt_masks).mean()
                cnts_c += 1

        print(f"Box APs: {format_array(box_ap_c.compute())}")
        print(f"Mask APs: {format_array(mask_ap_c.compute())}")

        if not supervised:
            acc = fg_acc_sum_c / cnts_c
            print(f"Foreground Acc: {acc:.4f}")

            fg_acc_sum += fg_acc_sum_c
            cnts += cnts_c

    print("All cell types:")
    print(f"Box APs: {format_array(box_ap.compute())}")
    print(f"Box APs: {format_array(mask_ap.compute())}")

    if not supervised:
        acc = fg_acc_sum / cnts
        print(f"Foreground Acc: {acc:.4f}")


if __name__ == "__main__":
    app()
