#!/usr/bin/env python

from pathlib import Path

# from joblib import Parallel, delayed
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import app, flags
from tqdm import tqdm
from skimage.transform import rescale

import lacss.data
import lacss.ops

from lacss.deploy import Predictor
from lacss.metrics import AP, LoiAP

# _CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_string("checkpoint", None, "", required=True)
flags.DEFINE_string("logpath", ".", "")
flags.DEFINE_float("nms", 0.0, "non-max-supress threshold")
flags.DEFINE_float("minscore", 0.2, "min score")
flags.DEFINE_string("datapath", "../../livecell_dataset/", "test data directory")
flags.DEFINE_float("minarea", 0.0, "min area of cells")
flags.DEFINE_float("dicescore", 0.5, "score threshold for Dice score")

FLAGS = flags.FLAGS

_th = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

_format = lambda x: ", ".join([f"{v:.4f}" for v in x])

avg_cell_sizes = {
    "A172": 34.6,
    "BT474": 24.6,
    "BV2": 13.3,
    "Huh7": 40.8,
    "MCF7": 18.5,
    "SHSY5Y": 20.1,
    "SKOV3": 44.8,
    "SkBr3": 21.8,
}

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
    import cv2
    pred_polygons = [ct.astype(int) for ct in pred["pred_contours"]]
    
    pred_bboxes = pred["pred_bboxes"]

    assert pred_bboxes.shape[-1] == 4

    intersects = np.zeros_like(box_its, dtype=int)

    gt_areas = np.array([len(mi) for mi in gt_m])
    areas = np.array([ cv2.contourArea(ct) for ct in pred_polygons])

    def _get_its(pred_id, gt_id):
        box= pred_bboxes[pred_id]
        polygon = pred_polygons[pred_id]
        mis = gt_m[gt_id]
        img = np.zeros([520, 704], dtype="uint8")
        cv2.fillPoly(img, [polygon], 1)
        return np.count_nonzero(
            img[(mis[:,0], mis[:,1])]
        )

    ids = np.where(box_its > 0)
    intersects[ids] = [_get_its(pid, gid) for pid, gid in zip(*ids)]

    return intersects, areas, gt_areas


def test_data():
    import glob
    import imageio.v2 as imageio
    from pycocotools.coco import COCO

    datapath = Path(FLAGS.datapath)
    annotation_file = datapath / "annotations"/"LIVECell"/"livecell_coco_test.json"
    image_path = datapath / "images"/"livecell_test_images"
    coco = COCO(annotation_file=annotation_file)

    for imgid in coco.getImgIds():
        bboxes, masks = [], []
        for ann_id in coco.getAnnIds(imgIds=imgid):
            ann = coco.anns[ann_id]
            bbox = ann["bbox"]
            bbox = np.array([bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]])
            bboxes.append(bbox)

            mask = coco.annToMask(ann)
            masks.append(np.stack(np.where(mask >= 0.5), axis=-1))

        bboxes = np.array(bboxes, dtype="float32")

        # remove redudnant
        n_boxes = bboxes.shape[0]
        selected = tf.image.non_max_suppression(
            bboxes, tf.ones([n_boxes], "float32"), n_boxes, iou_threshold=0.75
        ).numpy()
        bboxes = bboxes[selected]
        masks = [masks[k] for k in selected]

        assert bboxes.shape[0] == len(masks)

        name = coco.imgs[imgid]["file_name"]
        filepath = list(image_path.glob(f"**/{name}"))
        assert len(filepath) == 1
        image = imageio.imread(filepath[0])

        cell_type = name.split("_")[0]

        yield dict(
            image=image[..., None],
            bboxes=bboxes,
            masks=masks,
            celltype=cell_type,
        )

def main(_):
    model = Predictor(FLAGS.checkpoint)
    print(f"Model parameters loaded from {FLAGS.checkpoint}")

    model.module.detector.max_output = 2560
    model.module.detector.min_score = FLAGS.minscore

    print(model.module)

    # loi_ap = {"all": LoiAP([5, 2, 1])}
    mask_ap = {"all": AP(_th)}
    box_ap = {"all": AP(_th)}
    dice = {"all": Dice()}

    for data in tqdm(test_data()):
        t = data['celltype']
        image = data["image"]
        scale = 32 / avg_cell_sizes[t]
        img_h, img_w, _ = image.shape
        
        if not t in mask_ap:
            # loi_ap[t] = LoiAP([5, 2, 1])
            mask_ap[t] = AP(_th)
            box_ap[t] = AP(_th)
            dice[t] = Dice()

        target_sz = np.round(np.array([img_h, img_w]) * scale).astype(int).tolist()
        pred = model.predict(
            image,
            reshape_to=target_sz,
            min_area=FLAGS.minarea,
            score_threshold=FLAGS.minscore,
            nms_iou=FLAGS.nms,
            output_type="contour",
        )

        # compute ious matrix
        box_its, box_areas, gt_box_areas = get_box_its(pred, data["bboxes"])
        box_ious = box_its / (box_areas[:, None] + gt_box_areas - box_its + 1e-8)

        mask_its, areas, gt_areas = get_mask_its(pred, data['masks'], box_its)
        mask_ious = mask_its / (areas[:, None] + gt_areas - mask_its + 1e-8)

        scores = pred["pred_scores"]

        # Dice
        valid_predictions = scores >= FLAGS.dicescore
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
        # print("LOIAP: ", loi_ap[t].compute())
        print("BoxAP: ", box_ap[t].compute())
        print("MaskAP: ", mask_ap[t].compute())
        print("Dice: ", dice[t].compute())


if __name__ == "__main__":
    app.run(main)
