from __future__ import annotations

from functools import partial
from pathlib import Path
import ml_collections
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

import lacss.data

INPUT_PADDING = 768
IMG_SIZE = [544, 544]
CELL_SZ = 32

def _standardize(image):
    image = tf.image.per_image_standardization(image)
    if tf.shape(image)[-1] == 1:
        image = tf.repeat(image, 3, axis=-1)
    if tf.shape(image)[-1] == 2:
        image = tf.pad(image, [[0, 0], [0, 0], [0, 1]])

    return tf.ensure_shape(image, [None, None, 3])

def augment(x, crop_size=IMG_SIZE, s=None):
    image = x["image"] / tf.reduce_max(x["image"])
    gamma = tf.random.uniform([], 0.75, 1.25)
    image = tf.image.adjust_gamma(image, gamma)

    channels = tf.random.shuffle(tf.range(tf.shape(image)[-1]))
    image = tf.gather(image, channels, axis=-1) # ch shuffle

    x["image"] = _standardize(image)

    if s is None:
        box = x["bboxes"]
        max_dim = tf.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
        s = CELL_SZ / tf.reduce_mean(max_dim)

    min_scale, max_scale = s * 0.75, s * 1.25
    scaling_y = tf.random.uniform([], min_scale, max_scale)
    scaling_x = tf.random.uniform([], min_scale, max_scale)
    h, w  = crop_size
    crop_sz = ( 
        int(tf.round( float(h) / scaling_y )), 
        int(tf.round( float(w) / scaling_x )),
    )

    x = lacss.data.random_crop_or_pad(x, target_size=crop_sz, area_ratio_threshold=0.5)
    x = lacss.data.resize(x, target_size = crop_size)

    x = lacss.data.flip_up_down(x, p=0.5)
    x = lacss.data.flip_left_right(x, p=0.5)

    x['image'] = tf.ensure_shape(x['image'], crop_size + [3])

    if "masks" in x:
        return dict(
            image=x['image'], centroids=x['centroids'], bboxes=x['bboxes'], masks=x['masks']
        )
    else:
        return dict(
            image=x['image'], centroids=x['centroids'], bboxes=x['bboxes']
        )        

def check_cell_number(x):
    n_cells = tf.shape(x['centroids'])[0]
    return n_cells >=4 and n_cells < INPUT_PADDING

def format_train_data(x):
    n_cells = tf.shape(x['centroids'])[0]
    padding = INPUT_PADDING - n_cells
    locs = tf.pad(x['centroids'], [[0, padding], [0,0]], constant_values=-1.0)
    bboxes = tf.pad(x['bboxes'], [[0, padding], [0,0]], constant_values=-1.0)
    masks = tf.pad(x['masks'], [[0, padding], [0,0], [0,0]], constant_values=-1.0)
    return dict(image=x['image'], gt_locations=locs), dict(gt_bboxes=bboxes, gt_masks=masks)
    
def _get_ds(name, split="train"):
    return (
        tfds.load(name, split=split)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .repeat()
    )

def format_test_data(x, target_size=IMG_SIZE, s=None):
    import lacss.data

    x['image'] = _standardize(x["image"])

    if s is None:
        box = x["bboxes"]
        max_dim = tf.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
        s = 32 / tf.reduce_mean(max_dim)
    h, w  = IMG_SIZE
    crop_sz = ( 
        int(tf.round( float(h) / s )), 
        int(tf.round( float(w) / s )),
    )

    x = lacss.data.random_crop_or_pad(x, target_size=crop_sz, area_ratio_threshold=0.5)
    x = lacss.data.resize(x, target_size = target_size)

    return dict(image=x['image']), dict(
        gt_locations=x["centroids"],
        gt_bboxes=x["bboxes"],
        gt_masks=x["masks"],
    )

def get_config():
    from lacss.modules import Lacss

    config = ml_collections.ConfigDict()
    config.name = "train_2d"

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.steps = 200000
    config.train.finetune_steps = 50000
    config.train.validation_interval = 10000
    config.train.lr = 1e-4
    config.train.weight_decay = 0.05
    config.train.backbone_dropout = 0.4

    # config.train.freeze = ["backbone/ConvNeXt_0"]

    config.train.config = ml_collections.ConfigDict()
    config.train.config.detection_roi = 8
    config.train.config.similarity_score_scaling = 4
    config.train.config.n_labels_min = 4
    config.train.config.n_labels_max = 25

    labeled_train_data = (
        tf.data.Dataset.sample_from_datasets(
            [
                _get_ds("cellpose", split="train+cyto2"),
                _get_ds("livecell"),
                _get_ds("nips"),
                _get_ds("nuclei_net"),
                _get_ds("a431"), 
                _get_ds("ovules_2d")
            ],
            [.20, .40, .20, .1, .05, .05],
        )
        # .ragged_batch(4)
        # .map(lacss.data.mosaic, num_parallel_calls=tf.data.AUTOTUNE)
        # .map(partial(lacss.data.random_crop_or_pad, target_size=IMG_SIZE, area_ratio_threshold=0.5),
        #     num_parallel_calls=tf.data.AUTOTUNE)
        .map(
            partial(lacss.data.cutout, size=10, n=50),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .filter(check_cell_number)
        .map(format_train_data)
    )

    config.model = Lacss.get_preconfigued("default")
    config.model.detector.max_output = 1024

    config.data = ml_collections.ConfigDict()
    config.data.ds_train = labeled_train_data
    config.data.ds_val = dict(
        cellpose = tfds.load("cellpose", split="val").map(format_test_data),
        livecell = tfds.load("livecell", split="val").map(format_test_data),
        nips = tfds.load("nips", split="val").map(format_test_data),
    )
    config.data.batch_size = 8

    return config
