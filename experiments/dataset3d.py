from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tensorflow as tf
import jax
jnp = jax.numpy

DATAPATH = Path("/home/FCAM/jyu/datasets")

def _rescale_label_3d(label, target_shape, *, batchsize=5120):
    from skimage.measure import regionprops
    from lacss.ops import sub_pixel_samples

    D0, H0, W0 = label.shape
    D, H, W = target_shape
    scaling = (D/D0, H/H0, W/W0)

    new_label = np.zeros(target_shape, dtype=int)
    for rp in regionprops(label):
        box = rp.bbox
        target_box = np.array(box) * (scaling * 2)
        target_box_i = np.r_[np.floor(target_box[:3]), np.ceil(target_box[3:])].astype(int)
        # target_box_i = np.clip(target_box, 0, np.array(target_shape))
        indices = np.mgrid[
            target_box_i[0]:target_box_i[3],
            target_box_i[1]:target_box_i[4],
            target_box_i[2]:target_box_i[5],
        ]
        indices_r = (np.moveaxis(indices, 0, -1) + 0.5)/scaling
        indices_r = indices_r.reshape(-1, 3)
        n_ind = indices_r.shape[0]
        n_pad = (n_ind - 1)//batchsize*batchsize + batchsize - n_ind
        indices_r = np.pad(indices_r, [[0, n_pad], [0,0]])

        new_image = sub_pixel_samples(
            jnp.asarray(label==rp.label, dtype="float32"), 
            jnp.asarray(indices_r),
            edge_indexing=True,
        )
        new_image = new_image[:n_ind].reshape(indices[0].shape)
        new_image = (new_image >= 0.5).astype(int) * rp.label
        
        new_label[tuple(indices)] = np.maximum(new_label[tuple(indices)], new_image)
    
    return new_label

def _format_image(image, label, *, depth=16):
    import random
    # normalize
    image -= image.mean()
    image /= image.std() + 1e-6

    # 3-ch
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 2:
        image = np.pad(image, [[0, 0], [0, 0], [0, 1]])

    # pad z slices
    z_pad = depth-image.shape[0]
    z_pad_a = random.randint(0, z_pad)
    image = np.pad(
        image,
        [[z_pad_a, z_pad - z_pad_a], [0,0], [0,0], [0,0]],
    )
    label = np.pad(
        label,
        [[z_pad_a, z_pad - z_pad_a], [0,0], [0,0]],
    )

    return image, label

def _augment(image, label):
    import random

    image = image / image.max()
    image =  np.clip(image, 1e-6, 1)
    gamma = random.uniform(0.8, 1.2)
    image = image ** gamma

    if random.random() >= .5:
        image = image[:, :, ::-1]
        label = label[:, :, ::-1]
    if random.random() >= .5:
        image = image[:, ::-1, :]
        label = label[:, ::-1, :]
    if random.random() >= .5:
        image = image[::-1, :, :]
        label = label[::-1, :, :]

    if image.shape[-1] > 1:
        chs = list(range(image.shape[-1]))
        random.shuffle(chs)
        image = image[..., chs]

    return image, label

def _measure_label(label, *, n=128):
    from skimage.measure import regionprops

    bboxes = []
    locs = []
    for prop in regionprops(label):
        bboxes.append(prop["bbox"])
        locs.append(prop["centroid"])
    
    locs = np.stack(locs) + .5
    bboxes = np.asarray(bboxes, dtype="float32")

    # pad to fixed size
    n_pad = max(0, n - len(locs))
    locs = np.pad(
        locs[:n], [[0, n_pad],[0,0]],
        constant_values=-1,
    )
    bboxes = np.pad(
        bboxes[:n], [[0, n_pad],[0,0]],
        constant_values=-1,
    )

    return locs, bboxes

def mayu_dataset_generator():
    import czifile

    datapath = DATAPATH/"mayu"
    
    for label_file in datapath.glob("Experiment*.gt.tif"):
        name = (label_file.name).split(".")[0]
        img_file = datapath/(name + ".czi")

        imgs = czifile.imread(img_file).astype("float32")
        labels = imageio.imread(label_file)
        n_labels = labels.shape[0]
                                  
        imgs = imgs.squeeze()
        imgs = imgs[:n_labels]
        imgs = imgs / imgs.max(axis=(1,2,3), keepdims=True)

        # down size
        imgs = jax.image.resize(imgs, imgs.shape[:2] + labels.shape[-2:], "linear")
        imgs = imgs[..., None] # add ch

        for img, label in zip(imgs, labels):
            img, label = _augment(img, label)

            locs, bboxes = _measure_label(label)

            img, label = _format_image(img, label)

            yield {
                "image": img.astype("float32"),
                "gt_locations": locs.astype("float32"),
            }, {
                "gt_labels": label.astype("int32"),
            }

ds_mayu = (
    tf.data.Dataset.from_generator(
        mayu_dataset_generator,
        output_signature = (
            {
                "image": tf.TensorSpec([16, 256, 256, 3], dtype=tf.float32),
                "gt_locations": tf.TensorSpec([128, 3], dtype=tf.float32),
            }, 
            {
                "gt_labels": tf.TensorSpec([16, 256, 256], dtype=tf.int32),
            },
        )
    )
)


def c3dl_map_fn(x):
    image, label = x['image'], x['label']
    locs = x['centroids']
    if tf.random.uniform([]) >=.5:
        image = image[::-1]
        label = label[::-1]
        locs = tf.constant([30, 0, 0], dtype=locs.dtype) + locs * tf.constant([-1, 1, 1], dtype=locs.dtype)

    image = tf.image.per_image_standardization(image)
    image = tf.repeat(image, 3, axis=-1)

    return dict(image=image, gt_locations=locs), dict(gt_labels=label)

ds_c3dl = (
    tf.data.Dataset.load(str(DATAPATH/"ctc3d"/"c3dl.ds"))
    .map(c3dl_map_fn)
)

def n3dh_generator():
    datapath = DATAPATH/"ctc3d"/"Fluo-N3DH-CE"
    names = ("01", "02")
    n_time = 190

    for name in names:
        for i in range(n_time):
            img = imageio.imread(datapath/name/f"t{i:03d}.tif").astype("float32")
            label = imageio.imread(datapath/f"{name}_GT"/"TRA"/f"man_track{i:03d}.tif").astype(int)

            img = img[..., None]
            
            img, label = _augment(img, label)
            locs, _ = _measure_label(label)

            img, label = _format_image(img, label, depth=36)

            # rescale to training size
            target_size = [18, 258, 186]
            size_ratio = np.array(target_size) / img.shape[:3]
            locs = np.where(
                locs >= 0,
                locs * size_ratio,
                -1,
            )
            img = jax.image.resize(img, target_size + [3], "linear")

            # make the image a nice size
            img = img[1:-1, 1:-1, ...]
            img = np.pad(img, [[0,0], [0, 0], [0, 256-186], [0,0]])
            locs = locs - [1, 1, 0]

            yield {
                "image": img.astype("float32"),
                "gt_locations": locs.astype("float32"),
            }

ds_n3dh = tf.data.Dataset.from_generator(
    n3dh_generator,
    output_signature = (
        {
            "image": tf.TensorSpec([16, 256, 256, 3], dtype=tf.float32),
            "gt_locations": tf.TensorSpec([128, 3], dtype=tf.float32),
        }
    )
)


ds3d=ds_mayu
