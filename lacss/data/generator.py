import glob
import json

import imageio
import numpy as np
import tensorflow as tf

try:
    from pycocotools.coco import COCO
except:
    pass
from os.path import join

from skimage.measure import regionprops


# FIXME avoid dependency on COCO
def _coco_generator(annotation_file, image_path):
    coco = COCO(annotation_file=annotation_file)
    for imgid in coco.getImgIds():
        imginfo = coco.imgs[imgid]
        bboxes = []
        mis = []
        locs = []
        rls = []
        for ann_id in coco.getAnnIds(imgIds=imgid):
            ann = coco.anns[ann_id]
            bbox = ann["bbox"]
            bbox = np.array([bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]])
            bboxes.append(bbox)

            mask = coco.annToMask(ann)
            mi = np.stack(np.where(mask >= 0.5), axis=-1)
            mis.append(mi)
            rls.append(mi.shape[0])

            locs.append(mi.mean(axis=0))

        bboxes = np.array(bboxes, dtype="float32")
        locs = np.array(locs, dtype="float32")
        mis = np.concatenate(mis, dtype="int64")
        rls = np.array(rls, dtype="int64")

        filepath = glob.glob(join(image_path, "**", imginfo["file_name"]))
        assert len(filepath) == 1

        image = imageio.imread(filepath[0])
        if len(image.shape) == 2:
            image = image[:, :, None]
        image = (image / 255.0).astype("float32")

        binary_mask = np.zeros(image.shape[:2], "float32")
        binary_mask[tuple(mis.transpose())] = 1.0

        mis = tf.RaggedTensor.from_row_lengths(mis, rls)

        yield {
            "img_id": imgid,
            "image": image,
            "mask_indices": mis,
            "locations": locs,
            "binary_mask": binary_mask,
            "bboxes": bboxes,
            "filename": imginfo["file_name"],
        }


def dataset_from_coco_annotations(
    annotation_file, image_path, image_shape=[None, None, 3]
):
    return tf.data.Dataset.from_generator(
        lambda: _coco_generator(annotation_file, image_path),
        output_signature={
            "img_id": tf.TensorSpec([], dtype=tf.int64),
            "image": tf.TensorSpec(image_shape, dtype=tf.float32),
            "mask_indices": tf.RaggedTensorSpec([None, None, 2], tf.int64, 1),
            "locations": tf.TensorSpec([None, 2], dtype=tf.float32),
            "binary_mask": tf.TensorSpec([None, None], dtype=tf.float32),
            "bboxes": tf.TensorSpec([None, 4], dtype=tf.float32),
            "filename": tf.TensorSpec([], dtype=tf.string),
        },
    )


def _simple_generator(annotation_file, image_path, normalize=True):
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    for ann in annotations:
        image = imageio.imread(join(image_path, ann["image_file"]))
        if len(image.shape) == 2:
            image = image[:, :, None]
        # image = (image / 255.0).astype('float32')
        image = image.astype("float32")
        if normalize:
            image -= image.mean()
            image /= image.std()

        binary_mask = imageio.imread(join(image_path, ann["mask_file"]))
        if len(binary_mask.shape) == 3:
            binary_mask = binary_mask[:, :, 0]
        binary_mask = (binary_mask > 0).astype("float32")

        locations = np.array(ann["locations"]).astype("float32")

        yield {
            "img_id": ann["img_id"],
            "image": image,
            "binary_mask": binary_mask,
            "locations": locations,
        }


def dataset_from_simple_annotations(
    annotation_file, image_path, image_shape=[None, None, 3], **kwargs
):
    return tf.data.Dataset.from_generator(
        lambda: _simple_generator(annotation_file, image_path, **kwargs),
        output_signature={
            "img_id": tf.TensorSpec([], dtype=tf.int64),
            "image": tf.TensorSpec(image_shape, dtype=tf.float32),
            "locations": tf.TensorSpec([None, 2], dtype=tf.float32),
            "binary_mask": tf.TensorSpec([None, None], dtype=tf.float32),
        },
    )


def _img_mask_pair_generator(ds_files, normalize=True):
    for k, (img_file, mask_file) in enumerate(ds_files):
        img = imageio.imread(img_file).astype("float32")
        if normalize:
            img -= img.mean()
            img /= img.std()
        if len(img.shape) == 2:
            img = img[:, :, None]
        mask = imageio.imread(mask_file)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        binary_mask = (mask > 0).astype("float32")

        bboxes = []
        locs = []
        # mis = []
        # mi_lengths = []
        for prop in regionprops(mask):
            bboxes.append(prop["bbox"])
            locs.append(prop["centroid"])
            # mi = np.array(np.where(prop['image'])).transpose()
            # mi = mi + bboxes[-1][:2]
            # mi_lengths.append(mi.shape[0])
            # mis.append(mi)
        bboxes = np.array(bboxes, dtype="float32")
        locs = np.array(locs, dtype="float32")
        # mis = tf.RaggedTensor.from_row_lengths(np.concatenate(mis), mi_lengths)
        yield {
            "img_id": k,
            "image": img,
            "locations": locs,
            "binary_mask": binary_mask,
            "bboxes": bboxes,
            "label": mask,
            # 'mask_indices': mis
        }


def dataset_from_img_mask_pairs(
    imgfiles, maskfiles, image_shape=[None, None, 3], **kwargs
):
    return tf.data.Dataset.from_generator(
        lambda: _img_mask_pair_generator(zip(imgfiles, maskfiles), **kwargs),
        output_signature={
            "img_id": tf.TensorSpec([], dtype=tf.int64),
            "image": tf.TensorSpec(image_shape, dtype=tf.float32),
            # 'mask_indices': tf.RaggedTensorSpec([None, None, 2], tf.int64, 1),
            "locations": tf.TensorSpec([None, 2], dtype=tf.float32),
            "binary_mask": tf.TensorSpec([None, None], dtype=tf.float32),
            "bboxes": tf.TensorSpec([None, 4], dtype=tf.float32),
            "label": tf.TensorSpec([None, None], dtype=tf.int32),
        },
    )
