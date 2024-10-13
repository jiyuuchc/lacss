from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from xtrain import unpack_x_y_sample_weight

from ..ops import sub_pixel_samples, gather_patches, coords_of_patches
from .common import mean_over_boolean_mask, get_image_shape

def supervised_instance_loss(batch, prediction):
    """LACSS instance loss, supervised with segmentation label"""
    preds = prediction["predictions"]
    if not "segmentations" in preds:
        return 0

    inputs, labels, _ = unpack_x_y_sample_weight(batch)

    instance_mask = preds["segmentation_is_valid"]
    instance_logit = preds["segmentations"]
    n_patches = instance_logit.shape[0]

    if "gt_labels" in labels:  # labeled with image label
        gt_labels = labels["gt_labels"]
        if gt_labels.ndim == 2:
            gt_labels = gt_labels[None, :, :]
        locs = jnp.stack([ 
            preds["segmentation_z0_coord"],
            preds["segmentation_y0_coord"],
            preds["segmentation_x0_coord"],
        ], axis=-1)

        gt_patches = gather_patches(gt_labels, locs, instance_logit.shape[1:4])
        gt_patches = gt_patches == jnp.arange(1, n_patches+1)[:, None, None, None]
        gt_patches = gt_patches.astype("float32")

    elif "gt_masks" in labels:
        gt_segs = labels["gt_masks"][:n_patches]
        gt_bboxes = labels["gt_bboxes"][:n_patches]

        assert instance_logit.shape[1] == 1 or gt_segs.ndim == 4

        if gt_segs.ndim == 3:
            gt_segs = gt_segs[:, None, :, :]

        if gt_bboxes.shape[-1] == 4:
            gt_bboxes = jnp.c_[
                jnp.zeros([n_patches, 1], dtype=gt_bboxes.dtype), 
                gt_bboxes[:, :2], 
                jnp.ones([n_patches, 1], dtype=gt_bboxes.dtype),
                gt_bboxes[:, 2:],
            ]

        assert gt_segs.ndim == 4
        assert gt_bboxes.shape[-1] == 6
        assert gt_segs.shape[0] == n_patches, f"label length is {gt_segs.shape[0]}, but prediction length is {n_patches}"

        # compute rescaled coorinats in edge indexing
        (zc, yc, xc), cmask = coords_of_patches(preds, inputs['image'].shape[:-1])
        if "mask" in inputs:
            cmask = cmask & inputs["mask"][zc, yc, xc]
        z0, y0, x0, z1, y1, x1 = jnp.swapaxes(gt_bboxes, 0, 1)
        zc = zc + 0.5 - jnp.expand_dims(z0, (1,2,3))
        yc = yc + 0.5 - jnp.expand_dims(y0, (1,2,3))
        xc = xc + 0.5 - jnp.expand_dims(x0, (1,2,3))
        zc = zc / jnp.expand_dims(z1-z0, (1,2,3)) * gt_segs.shape[-3]
        yc = yc / jnp.expand_dims(y1-y0, (1,2,3)) * gt_segs.shape[-2]
        xc = xc / jnp.expand_dims(x1-x0, (1,2,3)) * gt_segs.shape[-1]

        # resample the label to match model coordinates
        gt_patches = jax.vmap(sub_pixel_samples)(
            gt_segs,
            jnp.stack([zc, yc, xc], axis=-1) -.5,
        ) >= 0.5
        gt_patches = jnp.where(cmask, gt_patches.astype("float32"), 0)

    else:
        return 0

    loss = optax.sigmoid_binary_cross_entropy(instance_logit, gt_patches)

    loss = loss.mean(axis=(1,2,3)).sum(where=instance_mask)

    return loss
    
    # return mean_over_boolean_mask(loss, instance_mask)

def instance_overlap_loss(batch, prediction, *, soft_label: bool = True):
    preds = prediction["predictions"]
    img_shape  = get_image_shape(batch)
    (cz, cy, cx), cmask = coords_of_patches(preds, img_shape)
    instances = jax.nn.sigmoid(preds['segmentations'])
    instances_sum = jax.numpy.zeros(img_shape)
    instance_mask = preds["segmentation_is_valid"]

    if soft_label:
        instances_sum = instances_sum.at[cz, cy, cx].add(jnp.where(cmask, instances, 0))
        loss = instances * jnp.where(cmask, instances_sum[cz, cy, cx] - instances, 1)
    else:
        log_yi = -jax.nn.log_sigmoid(-preds['segmentations'])
        instance_sum = instance_sum.at[cz, cy, cx].add(jnp.where(cmask, log_yi, 0))
        log_yi = jnp.where(
            cmask, 
            instance_sum[cz, cy, cx] - log_yi,
            1,
        )
        loss = instances * log_yi

    return mean_over_boolean_mask(loss, instance_mask)

def self_supervised_instance_loss(batch, prediction):
    """Unsupervised instance loss"""
    preds = prediction["predictions"]
    img_shape  = get_image_shape(batch)
    (cz, cy, cx), cmask = coords_of_patches(preds, img_shape)
    instances = jax.nn.sigmoid(preds['segmentations'])
    instance_mask = preds["segmentation_is_valid"]

    binary_mask = jax.lax.stop_gradient(jax.nn.sigmoid(preds["fg_pred"]))
    inputs, _, _ = unpack_x_y_sample_weight(batch)
    if 'mask' in inputs:
        binary_mask = binary_mask * inputs['mask']
    if binary_mask.ndim == 2:
        assert img_shape[0] == 1
        binary_mask = binary_mask[None, :, :]

    seg_patch = jnp.where(cmask, binary_mask[cz, cy, cx], 0)

    assert seg_patch.shape == instances.shape

    loss = (1 - seg_patch) * instances + seg_patch * (1 - instances)

    loss = mean_over_boolean_mask(loss, instance_mask)
    
    # loss += instance_overlap_loss(batch, prediction, soft_label=soft_label)

    return loss

def weakly_supervised_instance_loss(batch, prediction):
    """Instance loss supervised by image mask instead of instance masks"""
    preds = prediction["predictions"]
    img_shape  = get_image_shape(batch)
    (cz, cy, cx), cmask = coords_of_patches(preds, img_shape)
    instances = jax.nn.sigmoid(preds['segmentations'])
    instance_mask = preds["segmentation_is_valid"]

    loss = instance_overlap_loss(batch, prediction, soft_label=False)

    _, labels, _ = unpack_x_y_sample_weight(batch)
    if isinstance(labels, dict):
        seg = labels["gt_image_mask"].astype("float32")
    else:
        seg = labels.astype("float32")
    if seg.ndim == 2:
        assert img_shape[0] == 1
        seg = seg[None,:,:]
    seg_patch = jnp.where(cmask, seg[cz, cy, cx], 0)
    loss_ = (1.0 - seg_patch) * instances + seg_patch * (1.0 - instances)

    loss += mean_over_boolean_mask(loss_, instance_mask)    

    return loss

def segmentation_loss(batch, prediction, *, pretraining=False):
    _, labels, _ = unpack_x_y_sample_weight(batch)

    if labels is None:
        labels = {}

    if "gt_labels" in labels or "gt_masks" in labels:  # supervised
        return supervised_instance_loss(batch, prediction)
    elif "gt_image_mask" in labels:  # supervised by point + imagemask
        return weakly_supervised_instance_loss(batch, prediction)
    else:  # point-supervised
        return (
            self_supervised_instance_loss(batch, prediction)
            + instance_overlap_loss(batch, prediction, soft_label=not pretraining),
        )
