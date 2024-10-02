from __future__ import annotations

from functools import partial

import ml_collections
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

TAGET_LABEL_LEN = 1024
CROP_SIZE = 192

BASEMODEL = "/home/FCAM/jyu/proj_lacss/models/lacss_v3_2d_091912"

def _train_data_map(
    data,
    *,
    crop_size = CROP_SIZE,
    rescale_factor = 0.2,
):
    import lacss.data

    data['masks'] = data['cell_masks']
    del data['cell_masks']

    s = tf.random.uniform([3]) * rescale_factor * 2 + (1 - rescale_factor)

    axes = tf.random.shuffle(tf.range(3))
    data = lacss.data.transpose(data, axes=axes, p = .5)
    target_size = tf.cast(tf.round(s * crop_size), tf.int32)
    data = lacss.data.random_crop_or_pad(
        data, 
        target_size=target_size,
        area_ratio_threshold=.5,
        clip_boxes=False,
    )

    data = lacss.data.resize(data, target_size=(crop_size, crop_size, crop_size))

    data = lacss.data.flip_top_bottom(data, p=.5)
    data = lacss.data.flip_left_right(data, p=.5)
    data = lacss.data.flip_up_down(data, p=.5)
    
    return data


def check_cell_number(x):
    n_cells = tf.shape(x['centroids'])[0]
    return n_cells >=4 and n_cells < TAGET_LABEL_LEN


def format_train_data(x):
    n_cells = tf.shape(x['centroids'])[0]
    padding = TAGET_LABEL_LEN - n_cells
    locs = tf.pad(x['centroids'], [[0, padding], [0,0]], constant_values=-1.0)
    bboxes = tf.pad(x['bboxes'], [[0, padding], [0,0]], constant_values=-1.0)
    masks = tf.pad(x['masks'], [[0, padding], [0,0], [0,0], [0,0]], constant_values=-1.0)

    return dict(image=x['image'], gt_locations=locs, image_mask=x['image_mask']>=0.5), dict(gt_bboxes=bboxes, gt_masks=masks)


def format_test_data_3d(data, *, crop_size = CROP_SIZE):
    import lacss.data

    data['masks'] = data['cell_masks']
    del data['cell_masks']

    data = lacss.data.random_crop_or_pad(
        data, target_size=(crop_size,) * 3, 
        clip_boxes=False, 
        area_ratio_threshold=0.5,
    )

    return dict(
        image = data['image'],
        image_mask = data['image_mask'] >= 0.5,
    ),dict(
        gt_locations = data['centroids'],
        gt_bboxes = data['bboxes'],
    )


def get_config():
    from lacss.modules import Lacss
    from lacss.modules.lpn3d import LPN3D
    from lacss.modules.segmentor3d import Segmentor3D
    
    config = ml_collections.ConfigDict()
    config.name = "train_3d"

    config.train = dict(
        seed = 4242,

        steps = 50000,
        finetune_steps = 10000,
        validation_interval = 5000,
        lr = 1e-4,
        weight_decay = 0.05,

        freeze=["backbone"],
    )
    config.train.config = ml_collections.ConfigDict()
    config.train.config.max_training_instances = 32
    config.train.config.n_labels_max = 40
    config.train.config.n_labels_min = 6
    config.train.config.similarity_score_scaling = 8

    model = Lacss.get_default_model()
    model.detector_3d = LPN3D()
    model.segmentor_3d = Segmentor3D()
    config.model = model.get_config()

    ds_3d_train = (
        tfds.load("combined_3d", split="train")
        .map(_train_data_map, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(check_cell_number)
        .map(format_train_data)
        .repeat()
        # .batch(1)
        # .prefetch(1)
    )

    config.data = ml_collections.ConfigDict()
    config.data.ds_train = ds_3d_train
    config.data.ds_val = tfds.load("combined_3d", split="train").map(format_test_data_3d)
    config.data.batch_size = 1

    return config
