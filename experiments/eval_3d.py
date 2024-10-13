#!/usr/bin/env python

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import jax
import numpy as np
from absl import app, flags
import ml_collections

_FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", None, "model checkpoint")
flags.DEFINE_string("logpath", ".", "logging directory")
flags.mark_flag_as_required('checkpoint')
flags.DEFINE_string("datapath", "/home/FCAM/jyu/datasets", "data directory")

def get_config(DATAPATH):
    DATAPATH = Path(DATAPATH)
    config = ml_collections.ConfigDict()

    config.data = ml_collections.ConfigDict()
    config.data.ovules = dict(
        imgfiles = sorted((DATAPATH / "ovules_test/images/").glob("*") ),
        maskfiles = sorted((DATAPATH / "ovules_test/labels_with_ignore/").glob("*") ),
        scaling = [1, 0.625, 0.625],
    )
    config.data.organoid = dict (
        imgfiles = sorted((DATAPATH / "Mouse-Organoid-Cells-CBG/train/images/").glob("*"))[:20],
        maskfiles = sorted((DATAPATH / "Mouse-Organoid-Cells-CBG/train/masks/").glob("*"))[:20],
        scaling = [2.0, 0.33, 0.33],
    )
    config.data.platynereis = dict(
        imgfiles = sorted((DATAPATH / "Platynereis-Nuclei-CBG/train/images/").glob("*"))[:1],
        maskfiles = sorted((DATAPATH / "Platynereis-Nuclei-CBG/train/masks/").glob("*"))[:1],
        scaling = [3, 1, 1], 
    )
    config.data.platynereis_ish = dict(
        imgfiles = sorted((DATAPATH / "Platynereis-ISH-Nuclei-CBG/test/images/").glob("*")),
        maskfiles = sorted((DATAPATH /"Platynereis-ISH-Nuclei-CBG/test/masks/").glob("*")),
        scaling = [1.35, 1.35, 1.35],
    )
    # config.data.fly = dict(
    #     imgfiles = sorted((DATAPATH / "mayu/extra").glob("exp1*")),
    #     maskfiles = sorted((DATAPATH /"mayu/labels").glob("exp1*")),
    #     scaling = [6, 1, 1],
    # )

    config.lpn_nms_threshold = 8
    config.nms_iou = 0.3

    return config

def _count_mask_overlaps(pred_mask, coords, bucket_size=100000):
    from lacss.ops import sub_pixel_samples
    n_coords = coords.shape[0]
    n_pad = (n_coords - 1) // bucket_size * bucket_size + bucket_size - n_coords
    coords = np.pad(coords, [[0, n_pad], [0,0]], constant_values=-1)
    return np.count_nonzero(
        sub_pixel_samples(pred_mask, coords, edge_indexing=True) >= 0.5
    )

def compute_ious(predictions, gt_label):
    from skimage.measure import regionprops
    from lacss.ops import box_intersection, box_area
    
    pred_masks = (predictions["pred_masks"] >= 0.5).astype(float)
    pred_bboxes = predictions["pred_bboxes"]

    rps = regionprops(gt_label)
    gt_bboxes = np.stack([rp.bbox for rp in rps])
    gt_mask_vols = np.stack([rp.area for rp in rps])

    box_its = box_intersection(pred_bboxes, gt_bboxes)
    box_vols = box_area(pred_bboxes)
    mask_vols = (pred_masks>=.5).mean(axis=(1,2,3)) * box_vols #FIXME not exact value

    # gt_box_vols = box_area(gt_bboxes)
    # box_ious = box_its / (box_vols[:, None] + gt_box_vols - box_its + 1e-8)

    mask_its = np.zeros_like(box_its)
    its_threshold = (mask_vols[:, None] + gt_mask_vols) / 3
    pred_ids, gt_ids = np.where(box_its > its_threshold)

    logging.info(f"need to compute ious for {len(pred_ids)} pairs")

    for pred_id, gt_id in zip(pred_ids, gt_ids):
        pred_mask = pred_masks[pred_id]
        pred_bbox = pred_bboxes[pred_id]
        coords = rps[gt_id].coords + 0.5
        coords = (coords - pred_bbox[:3]) / (pred_bbox[3:] - pred_bbox[:3])
        coords = coords * pred_mask.shape
        mask_its[pred_id, gt_id] = _count_mask_overlaps(pred_mask, coords)

        # gt_mask_vol = rps[gt_id].area
        # mask_ious[pred_id, gt_id] = its/(gt_mask_vol + mask_vol - its + 1e-8)

    logging.info(f"done ious computation ")

    return mask_its, mask_vols, gt_mask_vols 


def main(_):
    import orbax.checkpoint as ocp
    from pprint import pp
    import imageio.v2 as imageio
    import pickle
    from lacss.deploy import Predictor
    from lacss.metrics import AP 
    from lacss.metrics.dice import Dice
    from lacss.utils import load_from_pretrained
    from tqdm import tqdm

    config = get_config(_FLAGS.datapath)

    def run_eval(model, params):
        model.detector_3d.nms_threshold = config.get("lpn_nms_threshold", 8)
        model.detector_3d.max_output=config.get("max_output", 512)

        predictor = Predictor((model, params))

        for name, data in config.data.items():
            print(f"====================== processing dataset {name}")
            scaling = data.scaling
            ap = AP([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
            dice = Dice()
            for imgfile, maskfile in zip(data.imgfiles, data.maskfiles):
                img = imageio.imread(imgfile)
                mask = imageio.imread(maskfile)
                if "Organoid" in str(imgfile):
                    img = img[:54]
                    mask = mask[:54]

                reshape_to = np.round(np.array(img.shape) * scaling).astype(int)

                logging.info(f"processing image {imgfile.stem} with shape {img.shape}")
                logging.info(f"predict at rescale shape {reshape_to}")

                predictions = predictor.predict_on_large_image(
                    img, reshape_to=reshape_to,
                    gs=256, ss=192, nms_iou=0.3, score_threshold=0.1,
                    output_type="bbox"
                )

                # remove masked cells
                img_mask = mask >= 0
                if not img_mask.all():
                    centroids = np.floor(predictions["pred_bboxes"].reshape(-1, 2, 3).mean(axis=1)).astype(int)
                    centroids = np.clip(centroids, 0, np.array(img_mask.shape) - 1)
                    bm = img_mask[tuple(centroids.transpose())]
                    predictions = jax.tree.map(lambda x: x[bm], predictions)

                n_cells = len(predictions["pred_scores"])
                n_gt_cells = len(np.unique(mask)) - 1
                logging.info(f"detected {n_cells} cells vs {n_gt_cells} cell in label")

                mask_its, mask_vols, gt_mask_vols = compute_ious(predictions, mask)
                mask_ious = mask_its / (mask_vols[:, None] + gt_mask_vols - mask_its + 1e-8)

                ap.update(mask_ious, predictions["pred_scores"])

                bm = predictions["pred_scores"] >= 0.2
                dice.update(mask_its[bm], mask_vols[bm], gt_mask_vols)
        
            print("MaskAP: ", ap.compute())
            print("Dice: ", dice.compute())

    cp = Path(_FLAGS.checkpoint)
    if not cp.is_dir():
        model, params = load_from_pretrained(cp)

        pp(model)

        run_eval(model, params)

    else:
        if (cp / "model.pkl").exists():
            print("seems to be a logging dir of multiple checkpoints. will eval all of them")
            all_cps = sorted(list(cp.glob("cp-*")))
            model_path = cp / "model.pkl"
        else:
            all_cps = [cp]
            model_path = cp.parent / "model.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        pp(model)

        for cp_path in all_cps:
            model, params= load_from_pretrained(cp_path)
            # params = ocp.StandardCheckpointer().restore(cp_path.absolute())["train_state"]["params"]        
            print(f"loaded parameters from {cp_path}")
            run_eval(model, params)


if __name__ == "__main__":
    jax.config.update("jax_compilation_cache_dir", "jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
    
    logging.basicConfig(level=logging.INFO)
    app.run(main)
