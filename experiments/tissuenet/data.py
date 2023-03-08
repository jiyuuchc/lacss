import io
import json
import os
import pickle
from functools import partial
from os.path import join

import jax
import numpy as np
import tensorflow as tf
from skimage.measure import regionprops

from lacss.data import parse_train_data_func


def pad_to(x, multiple=256):
    x = np.asarray(x)
    s = x.shape[0]
    ns = ((s - 1) // multiple + 1) * multiple
    padding = ns - s
    return np.pad(x, [[0, padding], [0, 0]], constant_values=-1.0)


_tissue_types = {
    "breast": 0,
    "gi": 1,
    "immune": 2,
    "lung": 3,
    "pancreas": 4,
    "skin": 5,
}


def train_dataset(data_path, supervised):
    X = np.load(join(data_path, "train", "X.npy"), mmap_mode="r+")
    Y = np.load(join(data_path, "train", "y.npy"), mmap_mode="r+")
    T = np.load(join(data_path, "train", "tissue_list.npy"), mmap_mode="r+")
    S = np.arange(len(X))
    np.random.shuffle(S)
    for k in S:
        image = X[k].astype("float32")
        label = Y[k][..., 0]
        locs = np.asarray([prop["centroid"] for prop in regionprops(label)])

        image = tf.image.random_contrast(image, 0.6, 1.4)
        image = tf.image.random_brightness(image, 0.3)

        if tf.random.uniform([]) >= 0.5:
            image = tf.image.transpose(image)
            locs = locs[..., ::-1]
            label = tf.transpose(label)

        if supervised:
            data = (
                dict(
                    image=image,
                    gt_locations=pad_to(locs),
                ),
                dict(mask_labels=label),
            )
        else:
            augmented = parse_train_data_func(
                dict(
                    image=image,
                    locations=locs,
                    binary_mask=np.asarray(label > 0).astype("float32"),
                ),
                size_jitter=[0.85, 1.15],
            )
            data = (
                dict(
                    image=augmented["image"].numpy(),
                    gt_locations=pad_to(augmented["locations"].numpy()),
                ),
                dict(
                    binary_mask=augmented["binary_mask"].numpy(),
                    tissue_type=np.asarray(_tissue_types[T[k]]),
                ),
            )

        data = jax.tree_util.tree_map(lambda v: np.expand_dims(v, 0), data)

        yield data


def val_dataset(data_path):
    X = np.load(join(data_path, "val", "X.npy"), mmap_mode="r+")
    Y = np.load(join(data_path, "val", "y.npy"), mmap_mode="r+")
    T = np.load(join(data_path, "val", "tissue_list.npy"), mmap_mode="r+")
    for k in range(len(X)):
        image = X[k].astype("float32")
        label = Y[k][..., 0]
        props = regionprops(label)
        locs = np.asarray([prop["centroid"] for prop in props])
        bboxes = np.asarray([prop["bbox"] for prop in props])
        data = (
            dict(image=image),
            dict(
                gt_boxes=pad_to(bboxes),
                gt_locations=pad_to(locs),
                tissue_type=np.asarray(_tissue_types[T[k]]),
            ),
        )
        data = jax.tree_util.tree_map(lambda v: np.expand_dims(v, 0), data)

        yield data
