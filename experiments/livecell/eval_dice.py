#!/usr/bin/env python

from pathlib import Path

import jax
import numpy as np
import typer
from tqdm import tqdm

import lacss
import lacss.deploy


def get_box_its(pred, gt_b):
    b = np.asarray(lacss.ops.bboxes_of_patches(pred))
    box_its = lacss.ops.box_intersection(gt_b, b)
    areas = lacss.ops.box_area(b)
    gt_areas = lacss.ops.box_are(gt_b)

    return box_its, areas, gt_areas


def get_mask_its(pred, gt_m, box_its):
    m = np.asarray(pred["instance_output"] >= 0.5)
    yc = np.asarray(pred["instance_yc"])
    xc = np.asarray(pred["instance_xc"])

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


class Dice:
    """Compute instance Dice values"""

    def __init__(self):
        self.pred_areas = []
        self.gt_areas = []
        self.pred_scores = []
        self.gt_scores = []

    def update(self, pred_its, pred_areas, gt_areas):
        self.pred_areas.append(pred_areas)
        self.gt_areas.append(gt_areas)

        pred_best = pred_its.max(axis=1)
        pred_best_matches = pred_its.argmax(axis=1)
        pred_dice = pred_best * 2 / (pred_areas + gt_areas[pred_best_matches])
        self.pred_scores.append(pred_dice)

        gt_best = pred_its.max(axis=0)
        gt_best_matches = pred_its.argmax(axis=0)
        gt_dice = gt_best * 2 / (gt_areas + pred_areas[gt_best_matches])
        self.gt_scores.append(gt_dice)

    def result(self):
        pred_areas = np.concatenate(self.pred_areas)
        gt_areas = np.concatenate(self.gt_areas)
        pred_scores = np.concatenate(self.pred_scores)
        gt_scores = np.concatenate(self.gt_scores)

        pred_dice = (pred_areas / pred_areas.sum() * pred_scores).sum()
        gt_dice = (gt_areas / gt_areas.sum() * gt_scores).sum()

        dice = (pred_dice + gt_dice) / 2

        return dice


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


def main(
    modelpath: Path,
    datapath: Path = Path("../../livecell_dataset"),
    logpath: Path = Path("."),
    nms: int = 8,
    min_score: float = 0.4,
    normalize: bool = False,
    min_area: int = 0,
):
    model = lacss.deploy.Predictor(modelpath, [520, 704, 1])

    model.detector = lacss.modules.Detector(
        test_nms_threshold=nms,
        test_max_output=3074,
        test_min_score=min_score,
    )

    print(model.module)

    test_data = lacss.data.coco_generator_full(
        datapath / "annotations/LIVECell/livecell_coco_test.json",
        datapath / "images/livecell_test_images",
    )

    dice = {"all": Dice()}
    for data in tqdm(test_data):
        t = data["filename"].split("_")[0]
        scale = cell_size_scales[t]

        image = data["image"]
        if normalize:
            image = (image - image.min()) / (image.max() - image.min())

        pred = model(
            (image,), remove_out_of_bound=True, min_area=min_area, scaling=1 / scale
        )

        box_its, _, _ = get_box_its(pred, data["bboxes"])
        mask_its, pred_areas, gt_areas = get_mask_its(pred, data["masks"] > 0, box_its)
        if not t in dice:
            dice[t] = Dice()
        dice[t].update(mask_its, pred_areas, gt_areas)
        dice["all"].update(mask_its, pred_areas, gt_areas)

    for t in sorted(dice.keys()):
        print(t)
        print(dice[t].result())


if __name__ == "__main__":
    typer.run(main)
