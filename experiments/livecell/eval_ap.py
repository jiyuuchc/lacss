#!/usr/bin/env python

from pathlib import Path

import jax
import numpy as np
import typer
from tqdm import tqdm

import lacss
import lacss.deploy
from lacss.metrics import AP, LoiAP

app = typer.Typer(pretty_exceptions_enable=False)

def get_ious(pred, gt_m, gt_b):
    b = np.asarray(lacss.ops.bboxes_of_patches(pred))
    m = np.asarray(pred["instance_output"] >= 0.5)
    yc = np.asarray(pred["instance_yc"])
    xc = np.asarray(pred["instance_xc"])

    box_its = lacss.ops.box_intersection(b, gt_b)
    b_area = lacss.ops.box_area(b)
    gt_b_area = lacss.ops.box_area(gt_b)
    box_ious = box_its / (b_area[:,None] + gt_b_area - box_its + 1e-8)
    
    pred_ids, gt_ids = np.where(box_its > 0)
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
    mask_ious = intersects / (areas[:,None] + gt_areas - intersects + 1e-8)

    return box_ious, mask_ious


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


@app.command()
def main(
    modelpath: Path,
    datapath: Path = Path("../../livecell_dataset"),
    logpath: Path = Path("."),
    nms: int = 4,
    min_area: int = 0,
    normalize: bool = False,
):
    model = lacss.deploy.Predictor(modelpath, [520,704,1])

    model.detector = lacss.modules.Detector(
        test_nms_threshold=nms,
        test_max_output=3074,
        test_min_score=0,
    )

    test_data = lacss.data.coco_generator_full(
        datapath / "annotations/LIVECell/livecell_coco_test.json",
        datapath / "images/livecell_test_images",
    )

    mask_ap = {"all": AP(_th)}
    box_ap = {"all": AP(_th)}
    loi_ap = {"all": LoiAP([5,2,1])}
    for data in tqdm(test_data):
        t = data["filename"].split("_")[0]
        scale = cell_size_scales[t]

        image = data["image"]
        h, w, _ = image.shape
        image = jax.image.resize(
            image, [round(h / scale), round(w / scale), 1], "linear"
        )
        if normalize:
            image = (image - image.min())/(image.max()-image.min())
        pred = model((image,), remove_out_of_bound=True, min_area=min_area)
        (
            pred["instance_output"],
            pred["instance_yc"],
            pred["instance_xc"],
        ) = lacss.ops.rescale_patches(pred, scale)
        pred["pred_locations"] = pred["pred_locations"] * scale

        box_ious, mask_ious = get_ious(
            pred, data["masks"] > 0, data["bboxes"]
        )
        
        scores = pred["pred_scores"]
        valid_predictions = pred['instance_mask'].squeeze() & (scores > 0)

        scores = scores[valid_predictions]
        mask_ious = mask_ious[valid_predictions]
        box_ious = box_ious[valid_predictions]

        if not t in mask_ap:
            mask_ap[t] = AP(_th)
            box_ap[t] = AP(_th)
            loi_ap[t] = LoiAP([5,2,1])
        mask_ap[t].update(mask_ious, scores)
        mask_ap["all"].update(mask_ious, scores)
        box_ap[t].update(box_ious, scores)
        box_ap["all"].update(box_ious, scores)
        loi_ap[t].update(pred, gt_locations=data["centroids"])
        loi_ap["all"].update(pred, gt_locations=data["centroids"])

    for t in sorted(mask_ap.keys()):
        print(t)
        print("LOIAP: ", loi_ap[t].compute())
        print("BoxAP: ", box_ap[t].compute())
        print("MaskAP: ", mask_ap[t].compute())


if __name__ == "__main__":
    app()
