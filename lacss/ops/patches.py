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

def gather_patches(source:ArrayLike, locations:ArrayLike, patch_size:int|tuple[int], *, padding_value=0):
    """ gather patches from an array

    Args:
        source: [D1..dk, ...]
        locations: [N, k] top-left coords of the patches, negative (out-of-bound) is ok
        patch_size: size of the patch. 
    Keyward Args:
        padding_value: value for out-of-bound locations

    returns:
        N patches [N, d1..dk, ...]
    """
    dim = locations.shape[-1]

    if locations.ndim != 2:
        raise ValueError(f"locations must be a 2d array, got shape {locations.shape}")

    if source.ndim < dim:
        raise ValueError(f"location has {dim} values, but the source data is less than {dim}-dim.")

    if isinstance(patch_size, int):
        patch_size = (patch_size,) * dim

    if len(patch_size) != dim:
        raise ValueError(f"length of patch_size does not match location shape.")

    src_shape = source.shape
    source = source.reshape(src_shape[:dim]+(-1,)) # [D1..Dk, B]

    slices = [slice(d) for d in patch_size]
    grid = jnp.mgrid[slices] + jnp.expand_dims(locations, axis=range(2, 2+dim)) # [N, k, d1..dk]
    grid = grid.swapaxes(0, 1) # [k, N, d1..dk]

    patches = source[tuple(grid)] # [N, d1..dk, B]
    max_d = jnp.expand_dims(jnp.asarray(src_shape[-dim-1:-1]), axis=range(1, dim+2))
    valid_loc = ((grid >= 0) & (grid<max_d)).all(axis=0) # [N, d1..dk]
    patches = jnp.where(
        valid_loc[..., None],
        patches,
        padding_value,       
    ) # clear out-of-bound locations

    patches = patches.reshape(patches.shape[:-1] + src_shape[dim:])

    return patches

def _get_patch_data(pred):
    if isinstance(pred, dict):
        patches = pred["segmentations"].squeeze()

        if "segmentation_y_coords" in pred:# backward compatibility
            yc = pred["segmentation_y_coords"]
            xc = pred["segmentation_x_coords"]
        else: 
            ps = patches.shape[-1]
            yc, xc = jnp.mgrid[:ps, :ps]
            yc = yc + pred["segmentation_y0_coord"][:, None, None]
            xc = xc + pred["segmentation_x0_coord"][:, None, None]
    else:
        patches, yc, xc = pred
        patches = patches.squeeze() #FIXME 

    # not true for stacks
    # assert patches.ndim == yc.ndim

    return patches, yc, xc


def bboxes_of_patches(pred: dict, threshold: float = 0, *, is2d:bool|None=None) -> jnp.ndarray:
    """Compute the instance bboxes from model predictions

    Args:
        pred: A model prediction dictionary:
        threshold: for segmentation, default 0

    Keyward Args:
        is2d: whether to force 2d output

    Returns:
        bboxes: [n, 4] or [n, 6]
    """
    patches = pred["segmentations"]
    y0 = pred["segmentation_y0_coord"]
    x0 = pred["segmentation_x0_coord"]

    n, d0, d1, d2 = patches.shape
    z_mask = jnp.any(patches > threshold, axis=(2,3))
    y_mask = jnp.any(patches > threshold, axis=(1,3))
    x_mask = jnp.any(patches > threshold, axis=(1,2))
    
    min_z = z_mask.argmax(axis=1)
    max_z = d0 - z_mask[:, ::-1].argmax(axis=1)

    min_y = y_mask.argmax(axis=1)
    max_y = d1 - y_mask[:, ::-1].argmax(axis=1)

    min_x = x_mask.argmax(axis=1)
    max_x = d2 - x_mask[:, ::-1].argmax(axis=1)

    min_y += y0
    max_y += y0
    min_x += x0
    max_x += x0

    if is2d is None:
        is2d = d0 == 1

    if is2d:
        bboxes = jnp.stack([min_y, min_x, max_y, max_x], axis=-1)
    else:
        bboxes = jnp.stack([min_z, min_y, min_x, max_z, max_y, max_x], axis=-1)

    is_valid = z_mask.any(axis=1, keepdims=True)
    bboxes = jnp.where(is_valid, bboxes, -1)

    return bboxes


# def patches_to_segmentations(
#     pred: DataDict, input_size: Shape, threshold: float = 0
# ) -> Array:
#     """Expand the predicted patches to the full image size.
#     The default model segmentation output shows only a small patch around each instance. This
#     function expand each patch to the size of the orginal image.

#     Args:
#         pred: A model prediction dictionary
#         input_size: shape of the input image. Tuple of H, W
#         threshold: for segmentation. Default is 0.5.

#     Returns:
#         segmentations: [n, height, width] n full0-size segmenatation masks.

#     """
#     patches, yc, xc = _get_patch_data(pred)
#     n_patches, patch_size, _ = yc.shape

#     page_nums = jnp.arange(n_patches)
#     segms = jnp.zeros((n_patches,) + input_size)
#     segms = segms.at[page_nums[:, None, None], yc, xc].set(patches)

#     return (segms >= threshold).astype(int)


def patches_to_label(
    pred: DataDict,
    input_size: Shape,
    *,
    mask: Optional[ArrayLike] = None,
    score_threshold: float = 0.5,
    threshold: float = 0,
) -> Array:
    """convert patch output to the image label.

    Args:
        pred: A model prediction dictionary
        input_size: shape of the input image. (H, W) or (D, H, W)
    Keyward Args:
        mask: boolean indicators masking out unwanted instances. Default is None (all cells)
        score_threshold: otional, min_score to be included. Default is .5.
        threshold: segmentation threshold.

    Returns:
        label: of the dimension input_size
    """
    input_size = tuple(input_size)
    if len(input_size) == 2:
        label = jnp.zeros((1,) + input_size)
    else:
        label = jnp.zeros(input_size)

    patches = pred["segmentations"]
    y0 = pred["segmentation_y0_coord"]
    x0 = pred["segmentation_x0_coord"]

    ps = patches.shape[-1]
    yy, xx = jnp.mgrid[:ps, :ps]
    yy += y0[:, None, None]
    xx += x0[:, None, None]

    if mask is None:
        mask = pred["segmentation_is_valid"]
    else:
        mask &= pred["segmentation_is_valid"]

    if score_threshold > 0:
        mask &= pred["scores"] >= score_threshold

    idx = jnp.cumsum(mask) * mask
    idx = jnp.where(mask, idx.max() + 1 - idx, 0)

    pr = (patches > threshold).astype(int) * idx[:, None, None, None]
    pr = pr.swapaxes(0,1) # [D, N, Ps, Ps]

    label = jax.vmap(lambda a, b: a.at[yy, xx].max(b))(label, pr)

    label = jnp.where(label, label.max() + 1 - label, 0)
    
    if len(input_size) == 2:
        label = label.squeeze(0)

    return label

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


def crop_and_resize_patches(pred:dict, target_shape:tuple[int] = (48,48), bboxes = None, *, threshold:float=0, convert_logits:bool=False):
    """ crop and rescale all instances to a target_size

    Args:
        pred: model predictions
        target_shape: output shape. usually a 3-tuple but can be a 2-tuple if input is a 2D image.
        bboxes: optionally supply bboxes for cropping
    Keyward Args:
        threshold: only used if bboxes is None. Threshold for computing the bboxes
        convert_logits: whether to convert the logits to probability
    
    Returns:
        Array [N] + target_shape.
    """
    from lacss.ops.image import sub_pixel_crop_and_resize
    patches = pred["segmentations"]
    y0 = pred["segmentation_y0_coord"]
    x0 = pred["segmentation_x0_coord"]

    if bboxes is None:
        bboxes = bboxes_of_patches(pred, threshold=threshold, is2d=False)
    elif bboxes.shape[-1] == 4:
        assert patches.shape[1] == 1, f"bboxes are for 2d but the patches are 3d"
        bboxes = jnp.c_[
            jnp.zeros_like(bboxes[:, 0]), 
            bboxes[:, :2],
            jnp.ones_like(bboxes[:, 0]), 
            bboxes[:, 2:],
        ]

    bboxes = bboxes.at[:,1].add(-y0)
    bboxes = bboxes.at[:,4].add(-y0)
    bboxes = bboxes.at[:,2].add(-x0)
    bboxes = bboxes.at[:,5].add(-x0)

    target_shape = tuple(target_shape)
    if len(target_shape) == 2:
        assert patches.shape[1] == 1, f"bboxes are for 2d but the patches are 3d"
        target_size_ = (1,) + target_shape
    else:
        target_size_ = target_shape
    
    if convert_logits:
        patches = jax.nn.sigmoid(patches)

    segs = jax.vmap(partial(sub_pixel_crop_and_resize, output_shape=target_size_))(
        patches,
        bboxes,
    )
    
    if len(target_shape) == 2:
        segs = segs.squeeze(1)
        
    return segs

