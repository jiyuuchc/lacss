""" Various functions deals with segmentation pathces

    All functions here takes unbatched input. Use vmap to convert to batched data
"""
from __future__ import annotations

from functools import partial
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..typing import *
from .boxes import box_iou_similarity
from .image import sub_pixel_samples

Shape = Sequence[int]

def _get_patch_data(pred):
    if isinstance(pred, dict):
        patches = pred["segmentations"].squeeze()

        if "segmentation_y_coords" in pred:
            yc = pred["segmentation_y_coords"]
            xc = pred["segmentation_x_coords"]
        else:
            ps = patches.shape[-1]
            yc, xc = jnp.mgrid[:ps, :ps]
            yc = yc + pred["segmentation_y0_coord"][:, None, None]
            xc = xc + pred["segmentation_x0_coord"][:, None, None]
    else:
        patches, yc, xc = pred
        patches = patches.squeeze()

    assert patches.ndim == yc.ndim

    return patches, yc, xc


def bboxes_of_patches(pred: Sequence | DataDict, threshold: float = 0) -> jnp.ndarray:
    """Compute the instance bboxes from model predictions

    Args:
        pred: A model prediction dictionary:
        threshold: for segmentation, default 0

    Returns:
        bboxes: [n, 4] bboox for empty patches are filled with -1
    """
    patches, yy, xx = _get_patch_data(pred)

    _, d0, d1 = patches.shape
    row_mask = jnp.any(patches > threshold, axis=1)
    col_mask = jnp.any(patches > threshold, axis=2)

    min_col = row_mask.argmax(axis=1)
    max_col = d1 - row_mask[:, ::-1].argmax(axis=1)

    min_row = col_mask.argmax(axis=1)
    max_row = d0 - col_mask[:, ::-1].argmax(axis=1)

    min_row += yy[:, 0, 0]
    max_row += yy[:, 0, 0]
    min_col += xx[:, 0, 0]
    max_col += xx[:, 0, 0]

    is_valid = row_mask.any(axis=1, keepdims=True)
    bboxes = jnp.stack([min_row, min_col, max_row, max_col], axis=-1)
    bboxes = jnp.where(is_valid, bboxes, -1)

    return bboxes


def patches_to_segmentations(
    pred: DataDict, input_size: Shape, threshold: float = 0
) -> Array:
    """Expand the predicted patches to the full image size.
    The default model segmentation output shows only a small patch around each instance. This
    function expand each patch to the size of the orginal image.

    Args:
        pred: A model prediction dictionary
        input_size: shape of the input image. Tuple of H, W
        threshold: for segmentation. Default is 0.5.

    Returns:
        segmentations: [n, height, width] n full0-size segmenatation masks.

    """
    patches, yc, xc = _get_patch_data(pred)
    n_patches, patch_size, _ = yc.shape

    page_nums = jnp.arange(n_patches)
    segms = jnp.zeros((n_patches,) + input_size)
    segms = segms.at[page_nums[:, None, None], yc, xc].set(patches)

    return (segms >= threshold).astype(int)


def patches_to_label(
    pred: DataDict,
    input_size: Shape,
    mask: Optional[ArrayLike] = None,
    score_threshold: float = 0.5,
    threshold: float = 0,
) -> Array:
    """convert patch output to the image label

    Args:
        pred: A model prediction dictionary
        input_size: shape of the input image. Tuple of H, W
        mask: boolean indicators masking out unwanted instances. Default is None (all cells)
        score_threshold: otional, min_score to be included. Default is .5.
        threshold: segmentation threshold.  Default .5

    Returns:
        label: [height, width]
    """
    label = jnp.zeros(input_size)
    patches, yy, xx = _get_patch_data(pred)

    pr = patches > threshold

    if mask is None:
        mask = pred["segmentation_is_valid"].squeeze(axis=(1, 2))
    else:
        mask &= pred["segmentation_is_valid"].squeeze(axis=(1, 2))

    if score_threshold > 0:
        mask &= pred["scores"] >= score_threshold

    idx = jnp.cumsum(mask) * mask
    idx = jnp.where(mask, idx.max() + 1 - idx, 0)

    pr = (pr > threshold).astype(int) * idx[:, None, None]

    label = label.at[yy, xx].max(pr)
    label = jnp.where(label, label.max() + 1 - label, 0)

    return label


# def ious_of_patches_from_same_image(pred: DataDict, threshold: float = 0.5) -> Array:
#     """Compute IOUs among instances from the same image. Most likely used for nms.

#     Args:
#         pred: A model prediction dictionary
#         threshold: Segmentation threshold. Default is 0.5.

#     Returns:
#         ious: IOUs as an upper triagle matrix.
#     """

#     bboxes = bboxes_of_patches(pred)
#     box_ious = box_iou_similarity(bboxes, bboxes)
#     box_ious = np.triu(box_ious, 1)
#     # n_detections = jnp.count_nonzero(pred['instance_mask'])
#     # box_ious = box_ious[:n_detections, :n_detections]

#     cy, cx = np.where(box_ious > 0)
#     pad_size = pred["instance_output"].shape[-1]

#     patches = np.asarray(pred["instance_output"]) >= threshold
#     patches_y = patches[cy]
#     patches_x = patches[cx]
#     plt_to = np.zeros([len(cy), pad_size * 3, pad_size * 3], dtype=bool)
#     dy = (
#         np.asarray(pred["instance_yc"])[cy, 0, 0]
#         - np.asarray(pred["instance_yc"])[cx, 0, 0]
#         + pad_size
#     )
#     dx = (
#         np.asarray(pred["instance_xc"])[cy, 0, 0]
#         - np.asarray(pred["instance_xc"])[cx, 0, 0]
#         + pad_size
#     )

#     for k in range(len(cy)):
#         plt_to[k, dy[k] : dy[k] + pad_size, dx[k] : dx[k] + pad_size] = patches_y[k]
#     plt_to = plt_to[:, pad_size : pad_size * 2, pad_size : pad_size * 2]
#     unions = np.count_nonzero(plt_to & patches_x, axis=(1, 2))

#     areas_y = np.count_nonzero(patches_y, axis=(1, 2))
#     areas_x = np.count_nonzero(patches_x, axis=(1, 2))

#     ious = unions / (areas_x + areas_y - unions + 1e-8)

#     mask_ious = box_ious
#     mask_ious[(cy, cx)] = ious

#     return mask_ious


# def non_max_suppress_predictions(pred: DataDict, iou_threshold: float = 0.6) -> Array:
#     """Perform nms on the model prediction based on mask IOU

#     Args:
#         pred: A model prediction dictionary
#         ious_threshold: default 0.6

#     Returns:
#         mask: boolean mask of cells not supressed
#     """

#     ious = ious_of_patches_from_same_image(pred)
#     mask = ious > iou_threshold

#     cnt = 0
#     while np.count_nonzero(mask) != cnt:
#         cnt = jnp.count_nonzero(mask)
#         can_suppress_others = ~mask.any(axis=0)
#         suppressed = (mask & can_suppress_others[:, None]).any(axis=0)
#         mask = mask & ~suppressed[:, None]

#     return ~mask.any(axis=0)


_sampling_op = jax.vmap(partial(sub_pixel_samples, edge_indexing=True))


def rescale_patches(
    pred: DataDict, scale: float, *, transform_logits: bool = True
) -> tuple[Array, Array, Array]:
    """Rescale/resize instance outputs in a sub-pixel accurate way.
    If the input image was rescaled, this function take care of rescaling the predictions
    to the orginal coodinates.

    Args:
        pred: A model prediction dictionary
        scale: The scaling value. The function does not take a noop shortcut even if scale is 1.

    Returns: A tuple of three arrays
        patches: a 3D array of the rescaled segmentation patches. The array shape should be different from
            the orignal patches in model predition.
        yy: The y-coordinates of the patches in mesh-grid format
        xx: The x-coordinates of the patches in mesh-grid format
    """
    patch, yc, xc = _get_patch_data(pred)
    if transform_logits:
        patch = jax.nn.sigmoid(patch)

    new_y0 = (yc[:, 0, 0] + 0.5) * scale  # edge indexing in new scale
    new_x0 = (xc[:, 0, 0] + 0.5) * scale  # edge indexing in new scale
    ps = round(patch.shape[-1] * scale)

    yy, xx = jnp.mgrid[:ps, :ps]
    yy = (
        jnp.floor(new_y0[:, None, None]).astype(int) + yy
    )  # center indexing in new scale
    xx = (
        jnp.floor(new_x0[:, None, None]).astype(int) + xx
    )  # center indexing in new scale

    # edge indexing in old scale
    rel_yy = (yy + 0.5) / scale
    rel_yy -= yc[:, :1, :1]
    rel_xx = (xx + 0.5) / scale
    rel_xx -= xc[:, :1, :1]

    new_patch = _sampling_op(
        patch,
        jnp.stack([rel_yy, rel_xx], axis=-1),
    )

    return new_patch, yy, xx
