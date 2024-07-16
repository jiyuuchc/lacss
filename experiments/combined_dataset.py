from __future__ import annotations

from pathlib import Path

import ml_collections
import tensorflow as tf

import lacss.data

INPUT_PADDING = 1024
IMG_SIZE = [544, 544]

data_path = Path("./datasets")

def _standardize(image):
    image = tf.image.per_image_standardization(image)
    if tf.shape(image)[-1] == 1:
        image = tf.repeat(image, 3, axis=-1)
    if tf.shape(image)[-1] == 2:
        image = tf.pad(image, [[0, 0], [0, 0], [0, 1]])

    return image

def augment(x, target_size=IMG_SIZE):
    box = x["bboxes"]
    max_dim = tf.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    s = 32 / tf.reduce_mean(max_dim)

    x = lacss.data.flip_up_down(x, p=0.5)
    x = lacss.data.flip_left_right(x, p=0.5)
    x = lacss.data.random_resize(x, scaling=[s * 0.7, s * 1.3])
    x = lacss.data.random_crop_or_pad(
        x, target_size=target_size, area_ratio_threshold=0.5
    )

    image = x["image"] / tf.reduce_max(x["image"])
    gamma = tf.random.uniform([], 0.8, 1.2)
    image = tf.image.adjust_gamma(image, gamma)

    channels = tf.random.shuffle(tf.range(tf.shape(image)[-1]))
    image = tf.gather(image, channels, axis=-1) # ch shuffle

    image = _standardize(image)

    return dict(
        image=tf.ensure_shape(image, target_size + [3]),
        gt_locations=x["centroids"][:INPUT_PADDING],
    ), dict(
        gt_bboxes=x["bboxes"][:INPUT_PADDING],
        gt_masks=x["masks"][:INPUT_PADDING],
    )


def scale_test_img(x, target_size=IMG_SIZE):
    box = x["bboxes"]
    max_dim = tf.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    scaling = 32 / tf.reduce_mean(max_dim)

    H = tf.cast(tf.cast(tf.shape(x["image"])[-3], tf.float32) * scaling + 0.5, tf.int32)
    W = tf.cast(tf.cast(tf.shape(x["image"])[-2], tf.float32) * scaling + 0.5, tf.int32)
    x = lacss.data.resize(x, target_size=[H, W])
    x = lacss.data.random_crop_or_pad(
        x, target_size=target_size, area_ratio_threshold=0.5
    )

    image = _standardize(x['image'])

    return dict(image=image,), dict(
        gt_locations=x["centroids"],
        gt_bboxes=x["bboxes"],
        gt_masks=x["masks"],
    )

ds_cellpose = (
    tf.data.Dataset.load(str(data_path / "cellpose.ds"), compression="GZIP")
    .map(augment)
    .repeat()
)

ds_livecell = (
    tf.data.Dataset.load(str(data_path / "livecell.ds"), compression="GZIP")
    .concatenate(
        tf.data.Dataset.load(str(data_path / "livecell_test.ds"), compression="GZIP")
    )
    .map(augment)
    .repeat()
)

ds_a431 = (
    tf.data.Dataset.load(
        str(data_path / "a431_label.ds"),
        compression="GZIP",
    )
    .map(augment)
    .repeat()
)

ds_combined = (
    tf.data.Dataset.sample_from_datasets(
        [ds_cellpose, ds_a431, ds_livecell],
        [0.25, 0.1, 0.65],
    )
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
)

ds_livecell_val = tf.data.Dataset.load(
    str(data_path / "livecell_val.ds"),
    compression="GZIP",
).map(scale_test_img)

ds_cellpose_val = tf.data.Dataset.load(
    str(data_path / "cellpose_test.ds"),
    compression="GZIP",
).map(scale_test_img)

def get_config():
    config = ml_collections.ConfigDict()

    config.dataset_name = "combined"

    config.ds_train = ds_combined
    config.ds_livecell_val = ds_livecell_val
    config.ds_cellpose_val = ds_cellpose_val

    return config
