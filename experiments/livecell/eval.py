#!/usr/bin/env python

from pathlib import Path

import jax
import numpy as np
import typer
from data import (
    cell_type_table,
    compress_masks,
    get_cell_type_and_scaling,
    remove_redundant,
)
from tqdm import tqdm

import lacss.data
import lacss.deploy
import lacss.ops
from lacss.metrics import AP, LoiAP

app = typer.Typer(pretty_exceptions_enable=False)

_format = lambda x: ", ".join([f"{v:.4f}" for v in x])


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

    def compute(self):
        pred_areas = np.concatenate(self.pred_areas)
        gt_areas = np.concatenate(self.gt_areas)
        pred_scores = np.concatenate(self.pred_scores)
        gt_scores = np.concatenate(self.gt_scores)

        pred_dice = (pred_areas / pred_areas.sum() * pred_scores).sum()
        gt_dice = (gt_areas / gt_areas.sum() * gt_scores).sum()

        dice = (pred_dice + gt_dice) / 2

        return dice


def get_box_its(pred, gt_b):
    b = pred["pred_bboxes"]
    box_its = lacss.ops.box_intersection(b, gt_b)
    areas = lacss.ops.box_area(b)
    gt_areas = lacss.ops.box_area(gt_b)

    return box_its, areas, gt_areas


def get_mask_its(pred, gt_m, box_its):
    m = pred["pred_masks"]
    mask_sz = m.shape[-1]

    yc, xc = np.mgrid[:mask_sz, :mask_sz]
    yc = yc + 1 + pred["pred_masks_y0"][:, None, None]
    xc = xc + 1 + pred["pred_masks_x0"][:, None, None]

    pred_ids, gt_ids = np.where(box_its > 0)

    gt_m = np.pad(gt_m, [[0, 0], [1, 1], [1, 1]])  # out-of-bound is 0
    _, h, w = gt_m.shape
    yc = np.clip(yc, 0, h - 1)
    xc = np.clip(xc, 0, w - 1)

    gt = gt_m[
        (
            gt_ids.reshape(-1, 1, 1),
            yc[pred_ids],
            xc[pred_ids],
        )
    ]

    v = np.count_nonzero(
        m[pred_ids] & gt,
        axis=(1, 2),
    )

    intersects = np.zeros_like(box_its, dtype="float32")
    intersects[(pred_ids, gt_ids)] = v

    areas = np.count_nonzero(m, axis=(1, 2))
    gt_areas = np.count_nonzero(gt_m, axis=(1, 2))

    return intersects, areas, gt_areas


_th = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


@app.command()
def main(
    modelpath: Path,
    datapath: Path = Path("../../livecell_dataset"),
    logpath: Path = Path("."),
    nms: int = 8,
    min_area: int = 0,
    min_score: float = 0.2,
    normalize: bool = True,
    dice_score: float = 0.5,
    v1_scaling: bool = False,
):
    model = lacss.deploy.Predictor(modelpath, [520, 704, 1])
    print(f"Model loaded from {modelpath.resolve()}")

    model.detector.test_nms_threshold = nms
    model.detector.test_max_output = 3074
    model.detector.test_min_score = min_score

    print(model.module)

    mask_ap = {"all": AP(_th)}
    box_ap = {"all": AP(_th)}
    # loi_ap = {"all": LoiAP([5, 2, 1])}
    dice = {"all": Dice()}

    test_data = (
        lacss.data.dataset_from_coco_annotations(
            datapath / "annotations/LIVECell/livecell_coco_test.json",
            datapath / "images/livecell_test_images",
            [520, 704, 1],
            None,
        )
        .map(remove_redundant)
        .map(compress_masks)
        .cache(str(datapath / "test_cache"))
    )
    print(test_data.element_spec)

    for data in tqdm(test_data):
        t, scale = get_cell_type_and_scaling(
            data["filename"], version=1 if v1_scaling else 2
        )
        t = list(cell_type_table.keys())[t]

        if not t in mask_ap:
            mask_ap[t] = AP(_th)
            box_ap[t] = AP(_th)
            # loi_ap[t] = LoiAP([5, 2, 1])
            dice[t] = Dice()

        # inference
        image = data["image"].numpy()
        if normalize:
            image = image - image.mean()
            image = image / image.std()

        pred = model.predict(
            image,
            # remove_out_of_bound=True,
            min_area=min_area,
            scaling=scale,
        )

        # transfer from GPU can be very slow. Try to minimize the amount
        bm = np.array(pred["instance_mask"].squeeze((1, 2)))
        pred = dict(
            pred_locations=np.array(pred["pred_locations"]),
            pred_bboxes=np.array(pred["pred_bboxes"]),
            pred_scores=np.array(pred["pred_scores"]),
            pred_masks=np.array(pred["instance_output"] >= 0.5),
            pred_masks_y0=np.array(pred["instance_yc"][:, 0, 0]),
            pred_masks_x0=np.array(pred["instance_xc"][:, 0, 0]),
        )
        pred = jax.tree_util.tree_map(lambda x: x[bm], pred)

        # recover masks from the compressed
        mi = data["masks"].numpy()
        n_instances = mi[:, 0].max() + 1
        img_h, img_w, _ = image.shape
        masks = np.zeros([n_instances, img_h, img_w], dtype="bool")
        masks[tuple(mi.transpose())] = True

        # compute ious matrix
        box_its, box_areas, gt_box_areas = get_box_its(pred, data["bboxes"].numpy())
        box_ious = box_its / (box_areas[:, None] + gt_box_areas - box_its + 1e-8)

        mask_its, areas, gt_areas = get_mask_its(pred, masks, box_its)
        mask_ious = mask_its / (areas[:, None] + gt_areas - mask_its + 1e-8)

        scores = np.asarray(pred["pred_scores"])

        # Dice
        valid_predictions = scores >= dice_score
        valid_its = mask_its[valid_predictions]
        valid_areas = areas[valid_predictions]

        dice[t].update(valid_its, valid_areas, gt_areas)
        dice["all"].update(valid_its, valid_areas, gt_areas)

        # various APs
        valid_ious = mask_ious
        valid_box_ious = box_ious
        valid_scores = scores

        mask_ap[t].update(valid_ious, valid_scores)
        mask_ap["all"].update(valid_ious, valid_scores)

        box_ap[t].update(valid_box_ious, valid_scores)
        box_ap["all"].update(valid_box_ious, valid_scores)

        # gt_locs = data["centroids"].numpy()
        # loi_ap[t].update(pred, gt_locations=gt_locs)
        # loi_ap["all"].update(pred, gt_locations=gt_locs)

    for t in sorted(mask_ap.keys()):
        print()
        print(t)
        # print("LOIAP: ", _format(loi_ap[t].compute()))
        print("BoxAP: ", _format(box_ap[t].compute()))
        print("MaskAP: ", _format(mask_ap[t].compute()))
        print("Dice: ", dice[t].compute())


if __name__ == "__main__":
    app()
