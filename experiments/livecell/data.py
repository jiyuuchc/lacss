import numpy as np
import tensorflow as tf

import lacss

cell_type_table = {
    "A172": 0,
    "BT474": 1,
    "BV2": 2,
    "Huh7": 3,
    "MCF7": 4,
    "SHSY5Y": 5,
    "SKOV3": 6,
    "SkBr3": 7,
}

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


def get_cell_type_and_scaling(name, version=2, *, target_size=34.6):
    if version == 1:
        return get_cell_type_and_scaling_v1(name)

    cell_type = name.numpy().split(b"_")[0].decode()
    cellsize = avg_cell_sizes[cell_type]
    cell_type_no = list(avg_cell_sizes.keys()).index(cell_type)
    return cell_type_no, target_size / cellsize


def get_cell_type_and_scaling_v1(name):
    cell_type = name.numpy().split(b"_")[0].decode()
    cell_type_no = list(cell_size_scales.keys()).index(cell_type)

    return cell_type_no, 1 / cell_size_scales[cell_type]


def remove_redundant(inputs):
    # remove redundant labels
    boxes = inputs["bboxes"]
    n_boxes = tf.shape(boxes)[0]
    selected = tf.image.non_max_suppression(
        boxes, tf.ones([n_boxes], "float32"), n_boxes, iou_threshold=0.75
    )

    inputs["bboxes"] = tf.gather(boxes, selected)
    inputs["masks"] = tf.gather(inputs["masks"], selected)
    inputs["centroids"] = tf.gather(inputs["centroids"], selected)

    return inputs


def compress_masks(inputs):

    inputs["masks"] = tf.where(inputs["masks"] > 0)

    return inputs


def augment(inputs, scaling=1.0, target_size=[544, 544]):
    image = inputs["image"]
    gamma = tf.random.uniform([], 0.5, 2.0)
    image = tf.image.adjust_gamma(image, gamma)
    inputs["image"] = tf.image.per_image_standardization(image)

    inputs = lacss.data.flip_up_down(inputs, p=0.5)
    inputs = lacss.data.flip_left_right(inputs, p=0.5)

    inputs = lacss.data.random_resize(inputs, scaling=[scaling * 0.8, scaling * 1.2])
    inputs = lacss.data.random_crop_or_pad(
        inputs, target_size=target_size, area_ratio_threshold=0.5
    )

    return inputs
