from __future__ import annotations

from pathlib import Path

import ml_collections
import tensorflow as tf

datapath = Path("/home/FCAM/jyu/datasets/mayu")

def dataset_generator():
    import imageio.v2 as imageio
    import czifile
    import jax
    import random
    import numpy as np
    from skimage.measure import regionprops
    
    for label_file in datapath.glob("Experiment*.gt.tif"):
        name = (label_file.name).split(".")[0]
        img_file = datapath /(name + ".czi")

        imgs = czifile.imread(img_file).astype("float32")
        labels = imageio.imread(label_file)
        n_labels = labels.shape[0]
                                  
        imgs = imgs.squeeze()
        imgs = imgs[:n_labels]
        imgs = imgs / imgs.max(axis=(1,2,3), keepdims=True)

        # down size
        imgs = jax.image.resize(imgs, imgs.shape[:2] + labels.shape[-2:], "linear")
        imgs = np.clip(imgs, 1e-6, 1)
        gamma = random.uniform(0.6, 1.4)
        imgs = imgs ** gamma

        # normalization
        imgs -= imgs.mean(axis=(1,2,3), keepdims=True)
        imgs /= imgs.std(axis=(1,2,3), keepdims=True) + 1e-6

        # pad z slices
        imgs = np.pad(
            imgs,
            [[0,0], [0, 16-imgs.shape[1]], [0, 0], [0, 0]],
        )
        labels = np.pad(
            labels,
            [[0,0], [0, 16-labels.shape[1]], [0, 0], [0, 0]],
        )

        for img, label in zip(imgs, labels):
            if random.random() >= .5:
                img = img[:, :, ::-1]
                label = label[:, :, ::-1]
            if random.random() >= .5:
                img = img[:, ::-1, :]
                label = label[:, ::-1, :]
            if random.random() >= .5:
                img = img[::-1, :, :]
                label = label[::-1, :, :]
            
            bboxes = []
            locs = []
            for prop in regionprops(label):
                bboxes.append(prop["bbox"])
                locs.append(prop["centroid"])
            
            locs = np.stack(locs) + .5
            bboxes = np.asarray(bboxes, dtype="float32")

            # pad to same length
            n_pad = 128 - locs.shape[0]
            locs = np.pad(locs, [[0, n_pad], [0, 0]], constant_values=-1)
            bboxes = np.pad(bboxes, [[0, n_pad], [0, 0]], constant_values=-1)

            # 3-ch
            img = np.repeat(img[..., None], 3, axis=-1)
            
            yield {
                "image": img.astype("float32"),
                "gt_locations": locs.astype("float32"),
            }, {
                "gt_labels": label.astype("int32"),
            }

ds3d = (
    tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature = (
            {
                "image": tf.TensorSpec([16, 256, 256, 3], dtype=tf.float32),
                "gt_locations": tf.TensorSpec([128, 3], dtype=tf.float32),
            }, 
            {
                "gt_labels": tf.TensorSpec([16,256,256], dtype=tf.int32),
            },
        )
    )
    .repeat()
)

def get_config():
    config = ml_collections.ConfigDict()

    config.dataset_name = "mayu"

    config.ds_train = ds3d

    return config
