#!/usr/bin/env python3

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from os.path import join
from pathlib import Path

import numpy as np
from tqdm import tqdm

import lacss
import lacss.deploy
from lacss.metrics.ranked import _unique_location_matching

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
    checkpoint: Path,
    datapath: Path = Path("../../livecell_dataset"),
    logpath: Path = Path("./"),
    min_area: float = 0,
    score_threshold: float = -1.0,
):
    print("evaluating...")

    predictor = lacss.deploy.Predictor(checkpoint)

    print(f"checkpoint loaded from {checkpoint}")

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    mask_ap = lacss.metrics.AP(thresholds)
    box_ap = lacss.metrics.AP(thresholds)

    tpa, fpa, fna = 0, 0, 0
    if score_threshold < 0:
        score_threshold = (0.4, 0.45, 0.5, 0.55, 0.6)
    else:
        score_threshold = (score_threshold,)

    for c in range(8):
        print(f" Cell type : {c}")

        test_data = data.test_data(
            datapath,
            supervised=True,
            cell_type=c,
        )

        mask_ap_c = lacss.metrics.AP(thresholds)
        box_ap_c = lacss.metrics.AP(thresholds)

        tp = np.zeros_like(score_threshold)
        fp = np.zeros_like(score_threshold)
        fn = np.zeros_like(score_threshold)

        for inputs, labels in tqdm(test_data()):
            pred = predictor(inputs, remove_out_of_bound=True, min_area=min_area)
            scores = np.asarray(pred["pred_scores"])
            valid_preds = scores > 0

            # boxes
            pred_boxes = np.asarray(lacss.ops.bboxes_of_patches(pred))
            gt_boxes = np.asarray(labels["gt_boxes"])
            box_ious = np.asarray(lacss.ops.box_iou_similarity(pred_boxes, gt_boxes))

            valid_box_ious = box_ious[valid_preds]
            box_ap_c.update(valid_box_ious, scores[valid_preds])
            box_ap.update(valid_box_ious, scores[valid_preds])

            # masks
            gt_mask_indices = np.asarray(labels["gt_mask_indices"])
            m_ious = mask_ious(pred, gt_mask_indices, box_ious)
            valid_m_ious = m_ious[valid_preds]
            mask_ap_c.update(valid_m_ious, scores[valid_preds])
            mask_ap.update(valid_m_ious, scores[valid_preds])

            # indicators = (valid_m_ious).max(axis=1) >= .5
            _, indicators = _unique_location_matching(valid_m_ious, 0.5)
            for k, th in enumerate(score_threshold):
                sel = scores[valid_preds] >= th
                cnts = indicators[sel].sum()
                tp[k] += cnts
                fp[k] += sel.sum() - cnts
                fn[k] += len(gt_boxes) - cnts

        print(f"Box APs: {format_array(box_ap_c.compute())}")
        print(f"Mask APs: {format_array(mask_ap_c.compute())}")
        print(f"TP={tp}, FP={fp}, FN={fn}")

        tpa = tp + tpa
        fpa = fp + fpa
        fna = fn + fna

    print("All cell types:")
    print(f"Box APs: {format_array(box_ap.compute())}")
    print(f"Mask APs: {format_array(mask_ap.compute())}")
    print(f"TP={tpa}, FP={fpa}, FN={fna}")


if __name__ == "__main__":
    app()
