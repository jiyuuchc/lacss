from __future__ import annotations

from pathlib import Path

import tensorflow as tf
import lacss.data

INPUT_PADDING = 1024
IMG_SIZE = [544, 544]
DATAPATH = Path("/home/FCAM/jyu/datasets")

def augment(x, target_size=IMG_SIZE):
    box = x["bboxes"]
    max_dim = tf.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    s = 32 / tf.reduce_mean(max_dim)

    x = lacss.data.flip_up_down(x, p=0.5)
    x = lacss.data.flip_left_right(x, p=0.5)
    x = lacss.data.random_resize(x, scaling=[s * 0.8, s * 1.2])
    x = lacss.data.random_crop_or_pad(
        x, target_size=target_size, area_ratio_threshold=0.5
    )

    image = x["image"] / tf.reduce_max(x["image"])
    # image = tf.image.adjust_gamma(image, gamma)
    # image = tf.image.per_image_standardization(image)
    if tf.shape(image)[-1] == 1:
        image = tf.repeat(image, 3, axis=-1)

    return dict(
        image=tf.ensure_shape(image, target_size + [3]),
        gt_locations=x["centroids"][:INPUT_PADDING],
    ), dict(
        gt_bboxes=x["bboxes"][:INPUT_PADDING],
        gt_masks=x["masks"][:INPUT_PADDING],
    )


def scale_test_img(x, target_size=IMG_SIZE):
    scales = {
        b"BT474": 1.0279709015128078,
        b"A172": 0.6510693512457046,
        b"MCF7": 1.3191146187106495,
        b"BV2": 2.185142214106535,
        b"Huh7": 0.6579372636248555,
        b"SkBr3": 1.2877806031436272,
        b"SKOV3": 0.5145641841316791,
        b"SHSY5Y": 0.9212827130226745,
    }
    ks = tf.constant(list(scales.keys()))
    scales = tf.constant(list(scales.values()))
    celltype = tf.argmax(ks == x["celltype"])
    scaling = tf.gather(scales, celltype)

    H = tf.cast(tf.cast(tf.shape(x["image"])[-3], tf.float32) * scaling + 0.5, tf.int32)
    W = tf.cast(tf.cast(tf.shape(x["image"])[-2], tf.float32) * scaling + 0.5, tf.int32)
    x = lacss.data.resize(x, target_size=[H, W])
    x = lacss.data.random_crop_or_pad(
        x, target_size=target_size, area_ratio_threshold=0.5
    )

    image = tf.image.per_image_standardization(x["image"])
    if tf.shape(image)[-1] == 1:
        image = tf.repeat(image, 3, axis=-1)

    return dict(image=image,), dict(
        gt_locations=x["centroids"],
        gt_bboxes=x["bboxes"],
        gt_masks=x["masks"],
    )


def get_data(datapath=DATAPATH):
    livecell_train = (
        tf.data.Dataset.load(str(datapath / "livecell.ds"), compression="GZIP")
        .map(augment)
        .filter(lambda x, _: tf.size(x["gt_locations"]) > 2)
        .padded_batch(
            1,
            padded_shapes=(
                dict(
                    image=IMG_SIZE + [3],
                    gt_locations=(INPUT_PADDING, 2),
                ),
                dict(
                    gt_bboxes=(INPUT_PADDING, 4),
                    gt_masks=(INPUT_PADDING, 48, 48),
                ),
            ),
            padding_values=-1.0,
        )
        .unbatch()
        .repeat()
    )

    livecell_val = tf.data.Dataset.load(
        str(datapath / "livecell_val.ds"),
        compression="GZIP",
    ).map(scale_test_img)

    return livecell_train, livecell_val
