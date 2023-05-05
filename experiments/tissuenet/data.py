import io
import json
import os
import pickle
from functools import cached_property, partial
from os.path import join

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.measure import regionprops

from lacss.data import parse_train_data_func
from lacss.ops import patches_to_label
from lacss.train import TFDatasetAdapter

_tissue_types = {
    "breast": 0,
    "gi": 1,
    "immune": 2,
    "lung": 3,
    "pancreas": 4,
    "skin": 5,
}

_RAND_8 = np.asarray([1830, 1071, 1579, 1666, 535, 3098, 493, 3000])

to_label = jax.vmap(partial(patches_to_label, input_size=(256, 256)))


def _make_label_continous(m):
    m = m.astype(int)

    k = np.unique(m)
    v = np.asarray(range(len(k)))

    mapping_ar = np.zeros(k.max() + 1, dtype=int)
    mapping_ar[k] = v

    return mapping_ar[m]


def from_data_path(data_path, ch=0):
    X = np.load(join(data_path, "X.npy"), mmap_mode="r+")
    Y = np.load(join(data_path, "y.npy"), mmap_mode="r+")
    T = np.load(join(data_path, "tissue_list.npy"), mmap_mode="r+")
    for k in range(len(X)):
        image = X[k].astype("float32")
        label_cnts = np.count_nonzero(Y[k], axis=(0, 1))
        is_reversed = label_cnts[0] < label_cnts[1]
        label = Y[k][..., ch] if not is_reversed else Y[k][..., 1 - ch]
        label = _make_label_continous(label)
        props = regionprops(label)
        locs = np.asarray([prop["centroid"] for prop in props])
        bboxes = np.asarray([prop["bbox"] for prop in props])

        yield dict(
            image=image,
            locations=locs,
            bboxes=bboxes,
            binary_mask=np.asarray(label > 0).astype("float32"),
            label=label,
            tissue_type=np.asarray(_tissue_types[T[k]]).reshape(1),
        )


def tfds_from_data_path(data_path, imgsize=[512, 512, 2], ch=0):
    return tf.data.Dataset.from_generator(
        partial(from_data_path, data_path=data_path, ch=ch),
        output_signature=dict(
            image=tf.TensorSpec(imgsize),
            locations=tf.TensorSpec([None, 2]),
            bboxes=tf.TensorSpec([None, 4]),
            binary_mask=tf.TensorSpec(imgsize[:2]),
            label=tf.TensorSpec(imgsize[:2]),
            tissue_type=tf.TensorSpec([1]),
        ),
    )


def train_parser(data):
    tissue = data["tissue_type"]
    augmented = parse_train_data_func(data, size_jitter=[0.85, 1.15])

    image = augmented["image"]
    locs = augmented["locations"]
    # mask = augmented['binary_mask']

    image = tf.image.random_contrast(image, 0.6, 1.4)
    image = tf.image.random_brightness(image, 0.3)
    if tf.random.uniform([]) >= 0.5:
        image = tf.image.transpose(image)
        locs = locs[:, ::-1]
        # mask = tf.transpose(mask)

    return dict(
        image=image,
        gt_locations=locs,
        category=tissue,
    )


def val_parser(data):
    return (
        dict(
            image=data["image"],
            category=data["tissue_type"],
        ),
        dict(
            gt_boxes=data["bboxes"],
            gt_locations=data["locations"],
            gt_labels=data["label"],
        ),
    )


def val_parser_supervised(data):
    return (
        dict(
            image=data["image"],
        ),
        dict(
            gt_boxes=data["bboxes"],
            gt_locations=data["locations"],
            gt_labels=data["label"],
        ),
    )


def train_parser_supervsied(inputs):

    image = inputs["image"]
    gt_locations = inputs["locations"]
    mask_labels = tf.cast(inputs["label"], tf.float32)

    if tf.random.uniform([]) >= 0.5:
        image = tf.image.transpose(image)
        gt_locations = gt_locations[..., ::-1]
        mask_labels = tf.transpose(mask_labels)

    x_data = dict(
        image=image,
        gt_locations=gt_locations,
    )
    y_data = dict(
        gt_labels=mask_labels,
    )

    return x_data, y_data


def train_data(datapath, n_buckets, batchsize, ch=0):

    ds_train = tfds_from_data_path(join(datapath, "train"), ch=ch).cache(
        join(datapath, "train_cache")
    )
    ds_train = ds_train.repeat().map(train_parser)
    ds_train = ds_train.filter(lambda x: tf.size(x["gt_locations"]) > 0)

    ds_train = ds_train.bucket_by_sequence_length(
        element_length_func=lambda x: tf.shape(x["gt_locations"])[0],
        bucket_boundaries=list(np.arange(1, n_buckets + 1) * (2560 // n_buckets) + 1),
        bucket_batch_sizes=(batchsize,) * (n_buckets + 1),
        padding_values=-1.0,
        pad_to_bucket_boundary=True,
    )

    return TFDatasetAdapter(ds_train, steps=-1).get_dataset()


def train_data_supervised(datapath, n_buckets, batchsize, ch=0):

    ds_train = tfds_from_data_path(join(datapath, "train"), ch=ch).cache(
        join(datapath, "train_cache")
    )
    ds_train = ds_train.repeat().map(train_parser_supervsied)
    ds_train = ds_train.filter(lambda x, _: tf.size(x["gt_locations"]) > 0)
    ds_train = ds_train.bucket_by_sequence_length(
        element_length_func=lambda x, _: tf.shape(x["gt_locations"])[0],
        bucket_boundaries=list(np.arange(1, n_buckets + 1) * (2560 // n_buckets) + 1),
        bucket_batch_sizes=(batchsize,) * (n_buckets + 1),
        padding_values=-1.0,
        pad_to_bucket_boundary=True,
    )

    return TFDatasetAdapter(ds_train, steps=-1).get_dataset()


def val_data(datapath, n_buckets, batchsize, ch=0):
    ds_val = tfds_from_data_path(
        join(datapath, "val"), imgsize=[256, 256, 2], ch=ch
    ).cache(join(datapath, "val_cache"))
    ds_val = ds_val.map(val_parser)
    ds_val = ds_val.bucket_by_sequence_length(
        element_length_func=lambda _, y: tf.shape(y["gt_locations"])[0],
        bucket_boundaries=list(np.arange(1, n_buckets + 1) * (1024 // n_buckets) + 1),
        bucket_batch_sizes=(batchsize,) * (n_buckets + 1),
        padding_values=-1.0,
        pad_to_bucket_boundary=True,
        drop_remainder=True,
    )

    return TFDatasetAdapter(ds_val).get_dataset()


def val_data_supervised(datapath, n_buckets, batchsize, ch=0):
    ds_val = tfds_from_data_path(
        join(datapath, "val"), imgsize=[256, 256, 2], ch=ch
    ).cache(join(datapath, "val_cache"))
    ds_val = ds_val.map(val_parser_supervised)
    ds_val = ds_val.bucket_by_sequence_length(
        element_length_func=lambda _, y: tf.shape(y["gt_locations"])[0],
        bucket_boundaries=list(np.arange(1, n_buckets + 1) * (1024 // n_buckets) + 1),
        bucket_batch_sizes=(batchsize,) * (n_buckets + 1),
        padding_values=-1.0,
        pad_to_bucket_boundary=True,
        drop_remainder=True,
    )

    return TFDatasetAdapter(ds_val).get_dataset()


def pad_and_stack(locs):
    sz = max([len(l) for l in locs])
    psz = (sz - 1) // 256 * 256 + 256
    locs = np.asarray(
        [np.pad(l, [[0, psz - len(l)], [0, 0]], constant_values=-1.0) for l in locs]
    )
    return locs


def save_images(imgs, locs=None, **kwargs):
    outs = []
    for img in imgs:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img)
        ax.axis("off")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4).numpy()
        outs.append(image)
    return np.asarray(outs)


class Samples:
    def __init__(self, datapath):
        self.datapath = datapath

    @cached_property
    def _imgs(self):
        _X = np.load(join(self.datapath, "val", "X.npy"), mmap_mode="r+")
        return _X[_RAND_8]

    @cached_property
    def _locs(self):
        _Y = np.load(join(self.datapath, "val", "y.npy"), mmap_mode="r+")
        return pad_and_stack(
            [
                [prop["centroid"] for prop in regionprops(p)]
                for p in _Y[_RAND_8, :, :, 0]
            ]
        )

    @cached_property
    def _types(self):
        _T = np.load(join(self.datapath, "val", "tissue_list.npy"), mmap_mode="r+")
        return [_tissue_types[_T[k]] for k in _RAND_8]

    @cached_property
    def images(self):
        imgs = np.concatenate([self._imgs, np.zeros_like(self._imgs[..., :1])], axis=-1)
        return imgs[..., ::-1]

    def sample_predict(self, trainer):
        key = jax.random.PRNGKey(42)
        rngs = {"droppath": key, "augment": key}

        def _predict(img, loc, cat):
            return trainer.state.apply_fn(
                {"params": trainer.params},
                **{"image": img, "gt_locations": loc, "category": cat},
                training=True,
                rngs=rngs
            )

        preds = [
            jax.jit(_predict)(img, loc, cat)
            for img, loc, cat in zip(self._imgs, self._locs, self._types)
        ]
        preds = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *preds)

        return preds
