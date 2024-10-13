#!/usr/bin/env python

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

from absl import app, flags

flags.DEFINE_string("checkpoint", None, "", required=True)
flags.DEFINE_string("datapath", "/home/FCAM/jyu/datasets/", "test data directory")
flags.DEFINE_float("nms", 0.0, "non-max-supress threshold")
flags.DEFINE_float("minscore", 0.1, "min score")
flags.DEFINE_float("minarea", 0.0, "min area of cells")
flags.DEFINE_float("dicescore", 0.4, "score threshold for Dice score")
flags.DEFINE_bool("multiscale", True, "whether test with multi-scale")
flags.DEFINE_bool("large", False, "whether to use large_image predictor")

FLAGS = flags.FLAGS

_th = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

def test_data():
    data_path = Path(FLAGS.datapath)
    if data_path.exists():
        for img_file in (data_path/"images").glob("*"):
            img = imageio.imread(img_file)
            label_file = data_path/"labels"/f"{img_file.stem}.tiff"
            label = imageio.imread(label_file)

            yield img, label
    else:
        import tensorflow_datasets as tfds

        ds = tfds.load(FLAGS.datapath, split='val')

        for x in ds.as_numpy_iterator():
            yield x['image'], x['label']

def main(_):
    from lacss.metrics.common import compute_mask_its
    from lacss.metrics.dice import Dice
    from lacss.metrics import AP
    from lacss.deploy import Predictor
    from tqdm import tqdm
    from skimage.transform import resize

    import jax
    jax.config.update("jax_compilation_cache_dir", "jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)

    model = Predictor(FLAGS.checkpoint)
    print(f"Model parameters loaded from {FLAGS.checkpoint}")

    model.module.detector.max_output = 2048
    model.module.detector.min_score = FLAGS.minscore

    print(model.module)

    mask_ap = AP(_th)
    dice = Dice()

    if FLAGS.multiscale:
        scales = [0.4, 0.65, 1.0, 1.4, 2.0]
    else:
        scales = [1.0]

    for image, label in tqdm(test_data()):
        best = 0
        for s in scales:
            target_sz = np.round(np.array(image.shape[:2]) * s).astype(int)
            if label.shape[:2] != image.shape[:2]:
                image = resize(image, label.shape[:2])

            if label.max() < 2000 and not FLAGS.large:
                pred = model.predict(
                    image,
                    reshape_to=target_sz,
                    min_area=FLAGS.minarea,
                    score_threshold=FLAGS.minscore,
                    nms_iou=FLAGS.nms,
                    output_type="contour",
                )
            else:
                pred = model.predict_on_large_image(
                    image,
                    reshape_to=target_sz,
                    min_area=FLAGS.minarea,
                    score_threshold=FLAGS.minscore,
                    # nms_iou=FLAGS.nms,
                    output_type="contour",
                    gs=544, ss=480,
                )


            # compute ious matrix
            mask_its_, areas_, gt_areas_ = compute_mask_its(pred, label)
            mask_ious_ = mask_its_ / (areas_[:, None] + gt_areas_ - mask_its_ + 1e-8)
            scores_ = pred["pred_scores"]

            ap = AP([0.5])
            ap.update(mask_ious_, scores_)
            if ap.compute()[0.5] > best:
                best = ap.compute()[0.5]
                scores, areas, gt_areas = scores_, areas_, gt_areas_
                mask_ious, mask_its = mask_ious_, mask_its_

        #AP
        mask_ap.update(mask_ious, scores)

        # Dice
        valid_predictions = scores >= FLAGS.dicescore
        valid_its = mask_its[valid_predictions]
        valid_areas = areas[valid_predictions]

        dice.update(valid_its, valid_areas, gt_areas)


    print("MaskAP: ", mask_ap.compute())
    print("Dice: ", dice.compute())


if __name__ == "__main__":
    app.run(main)
