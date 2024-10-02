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


def bboxes_of_patches(pred: dict, threshold: float = 0, *, image_shape:tuple|None = None, is2d:bool|None=None) -> jnp.ndarray:
    """Compute the instance bboxes from model predictions

    Args:
        pred: A model prediction dictionary:
        threshold: for segmentation, default 0

    Keyward Args:
        image_shape: if not None, the boxes are clipped to be within the bound
        is2d: whether to force 2d output

    Returns:
        bboxes: [n, 4] or [n, 6]
    """
    patches = pred["segmentations"]
    z0 = pred["segmentation_z0_coord"]
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

    min_z += z0
    max_z += z0
    min_y += y0
    max_y += y0
    min_x += x0
    max_x += x0

    if is2d is None:
        is2d = d0 == 1

    if image_shape is not None:
        if is2d:
            d = 1
            h, w = image_shape
        else:
            d, h, w = image_shape
        min_z = np.clip(min_z, 0, d-1)
        min_y = np.clip(min_y, 0, h-1)
        min_x = np.clip(min_x, 0, w-1)
        max_z = np.clip(max_z, 1, d)
        max_y = np.clip(max_y, 1, h)
        max_x = np.clip(max_x, 1, w)

    if is2d:
        bboxes = jnp.stack([min_y, min_x, max_y, max_x], axis=-1)
    else:
        bboxes = jnp.stack([min_z, min_y, min_x, max_z, max_y, max_x], axis=-1)

    is_valid = z_mask.any(axis=1)
    is_valid &= (max_z > min_z) & (max_y > min_y) & (max_x > min_x)
    bboxes = jnp.where(is_valid[:, None], bboxes, -1)

    return bboxes


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

    if mask is None:
        mask = pred["segmentation_is_valid"]
    else:
        mask &= pred["segmentation_is_valid"]
    if score_threshold > 0:
        mask &= pred["scores"] >= score_threshold

    assert label.ndim == 3, f'invalid input_size {input_size}'
    assert mask.ndim == 1

    patches = pred["segmentations"]
    (zz, yy, xx), cmask = coords_of_patches(pred, label.shape)

    idx = jnp.cumsum(mask) * mask
    idx = jnp.where(mask, idx.max() + 1 - idx, 0)

    pr = (patches > threshold).astype(int) * idx[:, None, None, None]

    label = label.at[zz, yy, xx].max(jnp.where(cmask, pr, 0))

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


def crop_and_resize_patches(pred:dict, bboxes, *, target_shape:tuple[int,...] = (48,48), convert_logits:bool=False):
    """ crop and rescale all instances to a target_size

    Args:
        pred: model predictions
        bboxes: optionally supply bboxes for cropping
    Keyward Args:
        target_shape: output shape. usually a 3-tuple but can be a 2-tuple if input is a 2D image.
        convert_logits: whether to convert the logits to probability
    
    Returns:
        Array [N] + target_shape.
    """
    from lacss.ops.image import sub_pixel_crop_and_resize

    patches = pred["segmentations"]
    z0 = pred["segmentation_z0_coord"]
    y0 = pred["segmentation_y0_coord"]
    x0 = pred["segmentation_x0_coord"]

    if bboxes.shape[-1] == 4:
        assert patches.shape[1] == 1, f"bboxes are for 2d but the patches are 3d"
        bboxes = jnp.c_[
            jnp.zeros_like(bboxes[:, 0]), 
            bboxes[:, :2],
            jnp.ones_like(bboxes[:, 0]), 
            bboxes[:, 2:],
        ]

    bboxes = bboxes - jnp.c_[z0, y0, x0, z0, y0, x0] # relative to instance crops

    target_shape = tuple(target_shape)
    if len(target_shape) == 2:
        assert patches.shape[1] == 1, f"target shape is 2d but the patches are 3d"
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

# def merge_patches(
#         preds:dict, 
#         input_size:tuple[int,...], 
#         *, 
#         min_score:float=0.5, 
#         reduction:str="sum",
#         convert_logits:bool=False,
# )->Array:
#     instance_mask = preds["segmentation_is_valid"]
#     if min_score > 0:
#         instance_mask &= preds["scores"] >= min_score
#     if len(input_size) == 2:
#         input_size = (1,) + tuple(input_size)

#     instances = preds["segmentations"]
#     if convert_logits:
#         instances = jax.nn.sigmoid(instances) #[N, sz, sy, sx]
    
#     sz, sy, sx = instances.shape[-3:]
#     pz, py, px = sz//2+1, sy//2+1, sx//2+1
#     zc, yc, xc = jnp.mgrid[:sz, :sy, :sx]
#     zc = zc + preds["segmentation_z0_coord"][:, None, None, None] + pz
#     yc = yc + preds["segmentation_y0_coord"][:, None, None, None] + py
#     xc = xc + preds["segmentation_x0_coord"][:, None, None, None] + px

#     merged = jnp.zeros(input_size, dtype=instances.dtype)
#     merged = jnp.pad(merged, [[pz, pz], [py, py], [px, px]])

#     if reduction == "sum":
#         merged = merged.at[zc, yc, xc].add(instances)
#     elif reduction == "max":
#         merged = merged.at[zc, yc, xc].max(instances)
#     else:
#         raise ValueError(f"invalid reduction {reduction}. should be 'sum' | 'max'")

#     merged = merged[pz:pz+sz, py:py+sy, px:px+sx]

#     return merged


def coords_of_patches(preds:dict, image_shape:tuple[int, ...])->tuple[Array, Array]:
    """ get the zyx coordinates of segmentations

    Args:
        preds: the model prediction dictionary
        image_shape: the original input image shape. (H, W) for 2d and (D, H, W) for 3d

    returns: a tuple of (coordinates, boolean_masks)
        coordinates: integer tensor of shape: (3,) + preds['segmentations'].shape
        boolan_masks: boolean tensor indicating whether the coordinate is a real one of padding.
    """
    locs = jnp.stack([ 
        preds["segmentation_z0_coord"],
        preds["segmentation_y0_coord"],
        preds["segmentation_x0_coord"],
    ]) #[3, N]

    sz, sy, sx = preds['segmentations'].shape[1:4] 
    if len(image_shape) == 2:
        assert sz == 1, f"2d image_shape but 3d predictions"
        image_shape = (1,) + tuple(image_shape)

    coords = jnp.mgrid[:sz, :sy, :sx] # [3, sz, sy, sx]
    coords = coords[:, None, ...] + locs[..., None, None, None] #[3, N, sz, sy, sx]
    mask = (coords >= 0).all(axis=0) & (jnp.moveaxis(coords, 0, -1) < jnp.asarray(image_shape)).all(axis=-1)
    mask = mask & preds["segmentation_is_valid"][:, None, None, None]

    return coords, mask
