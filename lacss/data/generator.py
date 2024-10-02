from __future__ import annotations

import glob
import json
import warnings
from functools import partial
from os.path import join
from pathlib import Path
from typing import Iterator, Optional, Sequence

import imageio.v2 as imageio
import numpy as np
import tensorflow as tf
from skimage.measure import regionprops


# A adaptation of tf.image.crop_and_resize to use edge-based indexing
# Edge-based indexing is more accurate for float-value bboxes
# i.e. value (0, 0) refers to the top-left cornor (not center) of the top-left pixel
def _crop_and_resize(masks, boxes, target_shape):
    box_h = boxes[:, 2] - boxes[:, 0]
    box_w = boxes[:, 3] - boxes[:, 1]
    dh = box_h / target_shape[0] / 2
    dw = box_w / target_shape[1] / 2
    boxes = tf.stack([dh, dw, -dh, -dw], axis=-1) + boxes - 0.5
    H = tf.shape(masks)[1]
    W = tf.shape(masks)[2]
    segs = tf.image.crop_and_resize(
        masks[..., None],
        boxes / [H - 1, W - 1, H - 1, W - 1],
        tf.range(tf.shape(boxes)[0]),
        target_shape,
    )
    return tf.squeeze(segs, axis=-1)


def coco_generator_full(
    annotation_file: str,
    image_path: str,
    mask_shape: Optional[tuple[int, int]] = None,
) -> Iterator[dict]:
    """A generator function to produce coco-annotated data

    Args:
        annotation_file: Path to coco annotation files
        image_path: Path to image directory
        mask_shape: If supplied, all the instance segmentations will be croped and resized to the specifed size. Otherwise,
            the segmentations are uncropped (in original image size)

    Yields:
        A data dictionary with thse keys:

            - id: data id
            - filename: image filename
            - image: an array [H, W, C]
            - masks: segmentation masks. [N, H, W] or [N,] + mask_shape
            - centroids: yx format.
            - bboxes: y0x0y1x1 format.
            - label: an array [H, W] representing pixel labels of all instances.
    """
    from pycocotools.coco import COCO

    coco = COCO(annotation_file=annotation_file)
    for imgid in coco.getImgIds():
        imginfo = coco.imgs[imgid]
        bboxes = []
        locs = []
        masks = []
        # segs = []
        for ann_id in coco.getAnnIds(imgIds=imgid):
            ann = coco.anns[ann_id]
            bbox = ann["bbox"]
            bbox = np.array([bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]])
            bboxes.append(bbox)

            mask = coco.annToMask(ann)
            masks.append(mask)

            mi = np.stack(np.where(mask >= 0.5), axis=-1)
            locs.append(mi.mean(axis=0) + 0.5)

        bboxes = np.array(bboxes, dtype="float32")
        locs = np.array(locs, dtype="float32")
        masks = np.array(masks, dtype="float32")

        filepath = glob.glob(join(image_path, "**", imginfo["file_name"]))
        assert len(filepath) == 1
        image = imageio.imread(filepath[0])

        if len(image.shape) == 2:
            image = image[:, :, None]
        image = (image / 255.0).astype("float32")

        img_h, img_w, _ = image.shape
        bboxes = np.clip(
            bboxes / [img_h, img_w, img_h, img_w],
            0.0,
            1.0,
        ) * [img_h, img_w, img_h, img_w]
        bboxes = bboxes.astype("float32")

        locs = np.clip(
            locs / [img_h, img_w],
            0.0,
            1.0,
        ) * [img_h, img_w]
        locs = locs.astype("float32")

        if mask_shape is not None:
            segs = _crop_and_resize(masks, bboxes, mask_shape).numpy()
        else:
            segs = masks

        n_cells = masks.shape[0]
        masks = masks.astype("int32") * np.arange(1, n_cells + 1).reshape(n_cells, 1, 1)
        label = masks.max(axis=0).astype("int32")

        yield {
            "id": imgid,
            "filename": imginfo["file_name"],
            "image": image,
            "masks": segs,
            "centroids": locs,
            "bboxes": bboxes,
            "label": label,
        }


def coco_generator(annotation_file: str, image_path: str) -> Iterator[dict]:
    from pycocotools.coco import COCO

    coco = COCO(annotation_file=annotation_file)
    for imgid in coco.getImgIds():
        imginfo = coco.imgs[imgid]
        bboxes = []
        segs = []
        for ann_id in coco.getAnnIds(imgIds=imgid):
            ann = coco.anns[ann_id]
            bbox = ann["bbox"]
            bbox = [bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]]
            bboxes.append(bbox)

            seg = np.reshape(coco.anns[ann_id]["segmentation"], [-1, 2])
            segs.append(seg)

        filepath = glob.glob(join(image_path, "**", imginfo["file_name"]))
        assert len(filepath) == 1

        image = imageio.imread(filepath[0])

        if len(image.shape) == 2:
            image = image[:, :, None]
        image = (image / 255.0).astype("float32")

        yield {
            "id": imgid,
            "filename": imginfo["file_name"],
            "image": image,
            "bboxes": bboxes,
            "polygons": segs,
        }


def dataset_from_coco_annotations(
    annotation_file: str,
    image_path: str,
    image_shape: tuple = [None, None, 3],
    mask_shape: tuple = [48, 48],
) -> tf.Dataset:
    """Obtaining a tensowflow dataset from coco annotations. See [coco_generator_full()](./#lacss.data.generator.coco_generator_full)

    Args:
        annotation_file: Path to coco annotation files
        image_path: Path to image directory
        image_shape: The expect image shapes. Use None to represent variable dimensions.
        mask_shape: If supplied, all the instance segmentations will be croped and resized to the specifed size. Otherwise,
            the segmentations are uncropped (in original image size)

    Returns:
        A tensorflow dataset.

    """
    if mask_shape is None:
        mask_spec = tf.TensorSpec([None] + image_shape[:2], dtype=tf.float32)
    else:
        mask_spec = tf.TensorSpec([None] + mask_shape, dtype=tf.float32)

    return tf.data.Dataset.from_generator(
        lambda: coco_generator_full(annotation_file, image_path, mask_shape=mask_shape),
        output_signature={
            "id": tf.TensorSpec([], dtype=tf.int64),
            "filename": tf.TensorSpec([], dtype=tf.string),
            "image": tf.TensorSpec(image_shape, dtype=tf.float32),
            "masks": mask_spec,
            "centroids": tf.TensorSpec([None, 2], dtype=tf.float32),
            "bboxes": tf.TensorSpec([None, 4], dtype=tf.float32),
            "label": tf.TensorSpec(image_shape[:2], dtype=tf.int32),
        },
    )


def simple_generator(annotation_file: str, image_path: str) -> Iterator[dict]:
    """A simple generator function to produce image data labeled with points and image-level segmentaion.

    Args:
        annotation_file: Path to the json format annotation file.
        image_path: Path to the image directory.

    Yields:
        A data dictionary with thse keys:

            * img_id: data id
            * image: an array [H, W, C]
            * centroids: yx format.
            * image_mask: segmentation masks for the image. [H, W]
    """

    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    for k, ann in enumerate(annotations):
        image = imageio.imread(join(image_path, ann["image_file"]))
        locations = np.array(ann["locations"]).astype("float32")
        ndim = locations.shape[-1]
        if len(image.shape) == ndim:
            image = image[..., None]
        assert image.ndim == ndim + 1

        if image.dtype is np.dtypes.UInt8DType:
            image = (image / 255.0).astype("float32")
        else:
            image = image.astype("float32")

        if "img_id" in ann:
            img_id = ann["img_id"]
        else:
            img_id = k

        if "mask_file" in ann:
            binary_mask = imageio.imread(join(image_path, ann["mask_file"]))
            yield {
                "img_id": img_id,
                "image": image,
                "centroids": locations,
                "image_mask": binary_mask,
            }
        else:
            yield {
                "img_id": img_id,
                "image": image,
                "centroids": locations,
            }


def dataset_from_simple_annotations(
    annotation_file: str,
    image_path: str,
    image_shape: Sequence[int|None]=[None, None, 1],
    has_binary_mask: bool=False,
) -> tf.data.Dataset:
    """Obtaining a tensowflow dataset from simple annotatiion. See [simple_generator()](./#lacss.data.generator.simple_generator)

    Args:
        annotation_file: Path to the json format annotation file.
        image_path: Path to the image directory.
        image_shape: The expect image shapes. Use None to represent variable dimensions.
        has_binary_mask: whether the annotation specific a binary mask

    Returns:
        A tensorflow dataset object
    """
    ndim = len(image_shape) - 1
    output_signature = {
        "img_id": tf.TensorSpec([], dtype=tf.int64),
        "image": tf.TensorSpec(image_shape, dtype=tf.float32),
        "centroids": tf.TensorSpec([None, ndim], dtype=tf.float32),
    }
    if has_binary_mask:
        output_signature.update({
            "image_mask": tf.TensorSpec(image_shape[:-1], dtype=tf.float32),
        })

    return tf.data.Dataset.from_generator(
        lambda: simple_generator(annotation_file, image_path),
        output_signature=output_signature,
    )

def _to_masks(label, boxes, target_shape=(8,48,48)):
    from lacss.ops import sub_pixel_crop_and_resize
    n = boxes.shape[0]
    ids = np.unique(label)[1:]
    assert n==len(ids)

    masks = []
    for k in range(n):
        label_ = (label == ids[k]).astype("int32")
        masks.append(np.asarray(
            sub_pixel_crop_and_resize(label_, boxes[k], output_shape=target_shape)
        ))
    return np.stack(masks)

def img_mask_pair_generator(
    imgfiles: Sequence[str | Path],
    maskfiles: Sequence[str | Path],
    *,
    mask_shape:tuple[int]|None = None,
) -> Iterator[dict]:
    """A generator function to produce image data labeled with segmentation labels.
        In this case, one has paired input images and label images as files on disk.

    Args:
        imgfiles: List of file pathes to input image file.
        maskfiles: List of file pathes to label image file.

    Keyword Args:
        mask_shape: if not None, generate cropped masks at specified shape

    Yields:
        A data dictionary with thse keys:

            * img_id: data id
            * image: an array [H, W, C]
            * centroids: yx format.
            * bboxes: y0x0y1x1 format.
            * label: an array [H, W] representing pixel labels of all instances.
            * masks: if mask_shape is not None
    """
    for k, (img_file, mask_file) in enumerate(zip(imgfiles, maskfiles)):
        img = imageio.imread(img_file).astype("float32")
        mask = imageio.imread(mask_file)
        if mask.min() > 0:
            warnings.warn("Label image has nonzero min.")
            # mask -= mask.min()

        if len(img.shape) == 2:
            img = img[..., None]
        if img.shape[-1] > 3:
            if img.shape[0] <=3:
                img = np.moveaxis(img, 0, -1)
            else:
                img = img[..., None]

        if mask.shape[-1] == 1:
            mask = mask[..., 0]

        assert img.shape[:-1] == mask.shape

        bboxes = []
        locs = []
        for prop in regionprops(mask):
            bboxes.append(prop["bbox"])
            locs.append(prop["centroid"])
        
        locs = np.asarray(locs, dtype="float32") + 0.5
        bboxes = np.asarray(bboxes)
        
        if mask_shape is not None:
            mask_crops = _to_masks(mask, bboxes, target_shape=mask_shape)

        data = {
            "img_id": k,
            "image": img.astype("float32"),
            "centroids": locs,
            "bboxes": bboxes,
            "label": mask,
        }
        
        if mask_shape is not None:
            data["masks"] = mask_crops

        yield data


# def _bbox_from_label(label):
#     import numpy as np
#     from skimage.measure import regionprops

#     bboxes = np.asarray([r["bbox"] for r in regionprops(label)])
#     centroids = np.asarray([r["centroids"] for r in regionprops(label)])

#     return bboxes, centroids


def _mask_from_label(inputs, *, mask_shape=[48, 48]):
    label = inputs["label"]

    # if not "bboxes" in inputs or not "centroids" in inputs:
    #     inputs["bboxes"], inputs["centroids"] = tf.numpy_func(
    #         _bbox_from_label,
    #         [label],
    #         [tf.float32, tf.float32],
    #         False,
    #     )

    n_instances = tf.shape(inputs["bboxes"])[0]

    label_expanded = tf.cast(
        label == tf.range(1, n_instances + 1)[:, None, None],
        tf.int32,
    )

    inputs["masks"] = _crop_and_resize(
        label_expanded,
        inputs["bboxes"],
        mask_shape,
    )

    return inputs


def dataset_from_img_mask_pairs(
    imgfiles: Sequence[str | Path],
    maskfiles: Sequence[str | Path],
    *,
    image_shape: Sequence[int | None] = [None, None, 3],
    generate_masks: bool = False,
    mask_shape: tuple[int, int] = [48, 48],
) -> tf.Dataset:
    """Obtaining a tensowflow dataset from image/label pairs.
            See [img_mask_pair_generator()](./#lacss.data.generator.img_mask_pair_generator)

    Args:
        imgfiles: List of file pathes to input image file.
        maskfiles: List of file pathes to label image file.

    Keyword Args:
        image_shape: The expect image shapes. Use None to represent variable dimensions.
        generate_masks: Whether to convert label images to indiviudal instance masks. This
            should be True if your data augmentation pipeline include rescaling ops or cropping
            ops., becasue these ops does not recompute the label image.
        mask_shape: Only when generate_mask=True. The resolution of the generated masks.

    Returns:
        A tensorflow dataset object
    """
    ndim = len(image_shape) - 1
    if generate_masks:
        ds = tf.data.Dataset.from_generator(
            lambda: img_mask_pair_generator(imgfiles, maskfiles, mask_shape=mask_shape),
            output_signature={
                "img_id": tf.TensorSpec([], dtype=tf.int64),
                "image": tf.TensorSpec(image_shape, dtype=tf.float32),
                "centroids": tf.TensorSpec([None, ndim], dtype=tf.float32),
                "bboxes": tf.TensorSpec([None, ndim*2], dtype=tf.float32),
                "label": tf.TensorSpec(image_shape[:ndim], dtype=tf.int32),
                "masks": tf.TensorSpec([None] + list(mask_shape), dtype=tf.float32),
            },
        )
    else:
        ds = tf.data.Dataset.from_generator(
            lambda: img_mask_pair_generator(imgfiles, maskfiles),
            output_signature={
                "img_id": tf.TensorSpec([], dtype=tf.int64),
                "image": tf.TensorSpec(image_shape, dtype=tf.float32),
                "centroids": tf.TensorSpec([None, ndim], dtype=tf.float32),
                "bboxes": tf.TensorSpec([None, ndim*2], dtype=tf.float32),
                "label": tf.TensorSpec(image_shape[:ndim], dtype=tf.int32),
            },
        )

    return ds
