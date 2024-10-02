from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from xtrain import unpack_x_y_sample_weight

from ..ops import sorbel_edges, sorbel_edges_3d, coords_of_patches
from ..ops.patches import _get_patch_data
from .common import binary_focal_factor_loss, get_image_shape, mean_over_boolean_mask

EPS = jnp.finfo("float32").eps

# def _compute_edge(batch, prediction):
#     preds = prediction["predictions"]
#     instance_mask = preds["segmentation_is_valid"]
#     instance_logit = preds['segmentations']
#     instance_output = jax.nn.sigmoid(instance_logit)
#     img_shape  = get_image_shape(batch)
#     (zc, yc, xc), cmask = coords_of_patches(preds, img_shape)
#     cmask = cmask & instance_mask[:, None, None, None]

#     if img_shape[0] == 1: #2d
#         patch_edges = jnp.square(sorbel_edges(instance_output.squeeze(1)))
#         patch_edges = patch_edges[:, None, :, :]
#     else:
#         patch_edges = jnp.square(sorbel_edges(instance_output.reshape(-1, * instance_output.shape[2:])))
#         patch_edges = patch_edges.reshape(2, *instance_output.shape)
        
#     patch_edges = patch_edges.sum(axis=0) / 8
#     patch_edges = jnp.sqrt(jnp.clip(patch_edges, 1e-8, 1.0))  # avoid GPU error
#     combined_edges = jnp.zeros(img_shape, dtype=instance_logit.dtype)
#     combined_edges = combined_edges.at[zc, yc, xc].add(jnp.where(cmask, patch_edges, 0))
#     combined_edges = jnp.tanh(combined_edges)

#     return combined_edges

# def _compute_edge(batch, prediction):
#     preds = prediction["predictions"]
#     instance_logit = preds['segmentations']
#     instance_output = jax.nn.sigmoid(instance_logit)
#     img_shape  = get_image_shape(batch)
#     (zc, yc, xc), cmask = coords_of_patches(preds, img_shape)

#     if img_shape[0] == 1: #2d
#         patch_edges = jnp.square(sorbel_edges(instance_output.squeeze(1)) / 4)
#         patch_edges = patch_edges.sum(axis=0)
#         patch_edges = patch_edges[:, None, :, :]
#     else:
#         patch_edges = jnp.square(sorbel_edges_3d(instance_output) / 6)
#         patch_edges = patch_edges.mean(axis=0)
        
#     patch_edges = jnp.sqrt(jnp.clip(patch_edges, 1e-8, 1.0))  # avoid GPU error
#     combined_edges = jnp.zeros(img_shape, dtype=instance_logit.dtype)
#     combined_edges = combined_edges.at[zc, yc, xc].add(jnp.where(cmask, patch_edges, 0))
#     combined_edges = jnp.tanh(combined_edges)

#     return combined_edges

def _compute_edge(batch, prediction):
    preds = prediction["predictions"]
    instance_logit = preds['segmentations']
    instance_output = jax.nn.sigmoid(instance_logit)
    img_shape  = get_image_shape(batch)
    (zc, yc, xc), cmask = coords_of_patches(preds, img_shape)

    patch_edges = jnp.abs(sorbel_edges_3d(instance_output) / 6)
    combined_edges = jnp.zeros((3,) + img_shape, dtype=instance_logit.dtype)
    combined_edges = combined_edges.at[:, zc, yc, xc].add(jnp.where(cmask, patch_edges, 0))
    combined_edges = jnp.tanh(combined_edges)

    return combined_edges

def cks_boundry_loss(batch, prediction):
    """Cell border prediction consistency loss"""
    inputs, _, _ = unpack_x_y_sample_weight(batch)

    # lacss pred
    instance_edge = _compute_edge(batch, prediction)
    # if 'mask' in inputs:
    #     instance_edge = jnp.where(inputs['mask'], instance_edge, 0)

    # aux pred
    instance_edge_pred = jax.nn.sigmoid(prediction["predictions"]["edge_pred"])
    # instance_edge_pred = prediction["predictions"]["edge_pred"]
    if instance_edge_pred.ndim == 3:
        instance_edge_pred = instance_edge_pred[None, ]
        instance_edge = instance_edge[1:]

    instance_edge_pred = jnp.moveaxis(instance_edge_pred, -1, 0)
    assert instance_edge_pred.shape == instance_edge.shape

    if 'mask' in inputs:
        return optax.l2_loss(instance_edge_pred, instance_edge).mean(where=inputs['mask'])
    else:
        return optax.l2_loss(instance_edge_pred, instance_edge).mean()

def cks_segmentation_loss(batch, prediction, *, offset_sigma=10.0, offset_scale=2.0):
    """Image segmentation consistenct loss for the collaboraor model"""

    preds = prediction["predictions"]
    img_shape  = get_image_shape(batch)
    instance_mask = preds["segmentation_is_valid"]
    logits = jax.lax.stop_gradient(preds["segmentations"])

    ps_z, ps_y, ps_x = logits.shape[-3:]
    offset = np.moveaxis(np.mgrid[:ps_z, :ps_y, :ps_x], 0, -1)
    offset = offset - [(ps_z-1)/2, (ps_y-1)/2, (ps_x-1)/2]
    offset = offset * [3, 1, 1] #FIXME shoudl use variable z_scale
    offset = (offset ** 2).sum(axis=-1)
    offset = np.exp(- offset / 2 / offset_sigma / offset_sigma) * offset_scale

    logits = logits + jnp.asarray(offset)

    (cz, cy, cx), cmask = coords_of_patches(preds, img_shape)
    cmask = cmask & instance_mask[:, None, None, None]
    label = jnp.zeros(img_shape) - 1.0e6
    label = label.at[cz, cy, cx].max(jnp.where(cmask, logits, -1e6))

    fg_pred = jax.nn.sigmoid(preds["fg_pred"])
    fg_label = jax.nn.sigmoid(label)

    inputs, _, _ = unpack_x_y_sample_weight(batch)
    if 'mask' in inputs:
        fg_label = fg_label * inputs['mask']

    if fg_pred.ndim == 2:
        fg_pred = fg_pred[None, :, :]
    assert fg_pred.shape == fg_label.shape

    loss = (1 - fg_label) * fg_pred + fg_label * (1 - fg_pred)

    inputs, _, _ = unpack_x_y_sample_weight(batch)

    return loss.mean()


def aux_size_loss(batch, prediction, *, weight=0.01):
    """Auxillary loss to prevent model collapse"""
    preds = prediction["predictions"]
    inputs, labels, _ = unpack_x_y_sample_weight(batch)

    img_shape  = get_image_shape(batch)
    instance_mask = preds["segmentation_is_valid"]
    instances = jax.nn.sigmoid(preds["segmentations"])
    instance_mask = preds["segmentation_is_valid"]
    (cz, cy, cx), cmask = coords_of_patches(preds, img_shape)

    if labels is not None and ("gt_labels" in labels or "gt_masks" in labels):
        return None

    areas = jnp.sum(instances, axis=(-1, -2, -3), where=cmask)
    areas = jnp.clip(areas, EPS, 1e6)
    loss = jax.lax.rsqrt(areas / instances.shape[1]) * instances.shape[-1]
    loss = mean_over_boolean_mask(loss, instance_mask)

    return loss * weight


def supervised_segmentation_loss(batch, prediction):
    preds = prediction["predictions"]
    _, labels, _ = unpack_x_y_sample_weight(batch)

    if "gt_labels" in labels:

        mask = (labels["gt_labels"] > 0).astype("float32")

    else:

        mask = labels["gt_image_mask"].astype("float32")

    return optax.sigmoid_binary_cross_entropy(preds["fg_pred"], mask).mean()


def collaborator_segm_loss(batch, prediction, *, sigma, pi):
    preds = prediction["predictions"]
    _, labels, _ = unpack_x_y_sample_weight(batch)

    if not "fg_pred" in preds:
        return None

    if labels is None:
        labels = {}

    if "gt_image_mask" in labels or "gt_labels" in labels:
        return supervised_segmentation_loss(batch, prediction)
    else:
        return cks_segmentation_loss(
            batch, prediction, offset_sigma=sigma, offset_scale=pi
        )


def collaborator_border_loss(batch, prediction):
    preds = prediction["predictions"]
    _, labels, _ = unpack_x_y_sample_weight(batch)

    if not "edge_pred" in preds:
        return None

    if labels is not None and ("gt_labels" in labels or "gt_masks" in labels):
        return None

    else:
        return cks_boundry_loss(batch, prediction)
