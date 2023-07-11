from functools import partial
from os.path import join

import numpy as np
import tensorflow as tf
from skimage.measure import regionprops

_tissue_types = {
    "breast": 0,
    "gi": 1,
    "immune": 2,
    "lung": 3,
    "pancreas": 4,
    "skin": 5,
}


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
            centroids=locs,
            bboxes=bboxes,
            image_mask=np.asarray(label > 0).astype("float32"),
            label=label,
            tissue_type=np.asarray(_tissue_types[T[k]]).reshape(1),
        )


def tfds_from_data_path(data_path, imgsize=[512, 512, 2], ch=0):
    return tf.data.Dataset.from_generator(
        partial(from_data_path, data_path=data_path, ch=ch),
        output_signature=dict(
            image=tf.TensorSpec(imgsize),
            centroids=tf.TensorSpec([None, 2]),
            bboxes=tf.TensorSpec([None, 4]),
            image_mask=tf.TensorSpec(imgsize[:2]),
            label=tf.TensorSpec(imgsize[:2]),
            tissue_type=tf.TensorSpec([1]),
        ),
    )
