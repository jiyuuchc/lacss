#!/usr/bin/env python

from pathlib import Path

import jax
import numpy as np
import typer
from tqdm import tqdm

import lacss
import lacss.deploy
from lacss.metrics import AP


def mask_its(pred, gt_m, gt_b):
    b = np.asarray(lacss.ops.bboxes_of_patches(pred))
    m = pred["instance_output"] >= 0.5
    yc = np.asarray(pred["instance_yc"])
    xc = np.asarray(pred["instance_xc"])

    box_its = lacss.ops.box_intersection(gt_b, b)
    gt_ids, pred_ids = np.where(box_its > 0)

    gt_m = np.pad(gt_m, [[0, 0], [192, 192], [192, 192]])
    gt = gt_m[
        (
            gt_ids.reshape(-1, 1, 1),
            yc[pred_ids] + 192,
            xc[pred_ids] + 192,
        )
    ]

    v = np.count_nonzero(
        m[pred_ids] & gt,
        axis=(1, 2),
    )
    intersects = np.zeros((b.shape[0], gt_b.shape[0]))
    intersects[(pred_ids, gt_ids)] = v

    areas = np.count_nonzero(m, axis=(1, 2))
    gt_areas = np.count_nonzero(gt_m, axis=(1, 2))

    return intersects, areas, gt_areas


cell_size_scales = {
    "A172": 1.0,
    "BT474": 0.65,
    "BV2": 0.50,
    "Huh7": 1.30,
    "MCF7": 0.50,
    "SHSY5Y": 1.0,
    "SKOV3": 1.30,
    "SkBr3": 0.75,
}


_th = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def main(
    modelpath: Path,
    datapath: Path = Path("../../livecell_dataset"),
    logpath: Path = Path("."),
    nms: int = 4,
):
    model = lacss.deploy.Predictor(modelpath)

    model.detector = lacss.modules.Detector(
        test_nms_threshold=nms,
        test_max_output=3074,
        test_min_score=0,
    )

    test_data = lacss.data.coco_generator_full(
        datapath / "annotations/LIVECell/livecell_coco_test.json",
        datapath / "images/livecell_test_images",
    )

    ap = {"all": AP(_th)}
    for data in tqdm(test_data):
        t = data["filename"].split("_")[0]
        scale = cell_size_scales[t]

        image = data["image"]
        h, w, _ = image.shape
        image = jax.image.resize(
            image, [round(h / scale), round(w / scale), 1], "linear"
        )
        pred = model((image,), remove_out_of_bound=True)
        (
            pred["instance_output"],
            pred["instance_yc"],
            pred["instance_xc"],
        ) = lacss.ops.rescale_patches(pred, scale)

        pred_its, pred_areas, gt_areas = mask_its(
            pred, data["masks"] > 0, data["bboxes"]
        )
        ious = pred_its / (pred_areas[:, None] + gt_areas - pred_its + 1e-8)

        if not t in ap:
            ap[t] = AP(_th)
        ap[t].update(ious, pred["pred_scores"])
        ap["all"].update(ious, pred["pred_scores"])

    for t in sorted(ap.keys()):
        print(t)
        print(ap[t].result())


if __name__ == "__main__":
    typer.run(main)
