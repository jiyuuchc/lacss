from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from ..ops import sorbel_edges
from .common import binary_focal_factor_loss
from lacss.train.utils import unpack_x_y_sample_weight

EPS = jnp.finfo("float32").eps


def _compute_edge(instance_output, instance_yc, instance_xc, height, width):
    _, ps, _ = instance_yc.shape
    padding = ps // 2 + 1

    patch_edges = jnp.square(sorbel_edges(instance_output))
    patch_edges = (patch_edges[0] + patch_edges[1]) / 8.0
    patch_edges = jnp.sqrt(jnp.clip(patch_edges, 1e-8, 1.0))  # avoid GPU error
    # patch_edges = jnp.where(patch_edges > 0, jnp.sqrt(patch_edges), 0)
    combined_edges = jnp.zeros([height + padding * 2, width + padding * 2])
    combined_mask = jnp.zeros([height + padding * 2, width + padding * 2], dtype=bool)

    combined_edges = combined_edges.at[
        instance_yc + padding, instance_xc + padding
    ].add(patch_edges)
    combined_mask = combined_mask.at[instance_yc + padding, instance_xc + padding].set(
        True
    )

    combined_edges = combined_edges[
        padding : padding + height, padding : padding + width
    ]
    combined_mask = combined_mask[padding : padding + height, padding : padding + width]
    combined_edges = jnp.where(
        combined_mask,
        jnp.tanh(combined_edges),
        -1,
    )

    return combined_edges


def self_supervised_edge_loss(batch, prediction):
    """Cell border prediction consistency loss"""

    preds = prediction
    inputs, _, _ = unpack_x_y_sample_weight(batch)

    instance_output = preds["instance_output"]
    instance_yc = preds["instance_yc"]
    instance_xc = preds["instance_xc"]
    height, width = inputs["image"].shape[:2]
    instance_edge = _compute_edge(
        instance_output, instance_yc, instance_xc, height, width
    )
    instance_edge_pred = jax.nn.sigmoid(preds["edge_pred"])

    return optax.l2_loss(instance_edge_pred, instance_edge).mean(
        where=instance_edge >= 0
    )


def self_supervised_segmentation_loss(
    batch, prediction, *, offset_sigma=jnp.array([10.0]), offset_scale=jnp.array([2.0])
):
    """Image segmentation consistenct loss for the collaboraor model"""

    preds = prediction
    inputs, _, _ = unpack_x_y_sample_weight(batch)

    height, width = inputs["image"].shape[:2]
    ps = preds["instance_yc"].shape[-1]

    offset_sigma = jnp.asarray(offset_sigma).reshape(-1)
    offset_scale = jnp.asarray(offset_scale).reshape(-1)

    def _max_merge(pred):
        label = jnp.zeros([height + ps, width + ps]) - 1.0e6
        yc, xc = pred["instance_yc"], pred["instance_xc"]
        logit = pred["instance_logit"]

        if (
            offset_sigma.size > 1
            and "cls_id" in inputs
            and inputs["cls_id"] is not None
        ):
            c = inputs["cls_id"].astype(int).squeeze()
        else:
            c = 0

        sigma = offset_sigma[c]
        scale = offset_scale[c]
        offset = (jnp.mgrid[:ps, :ps] - (ps - 1) / 2) ** 2 / (2 * sigma * sigma)
        offset = jnp.exp(-offset.sum(axis=0)) * scale
        logit += offset

        label = label.at[yc + ps // 2, xc + ps // 2].max(logit)
        label = label[ps // 2 : -ps // 2, ps // 2 : -ps // 2]

        return label

    # fg_pred = jax.nn.tanh(preds["fg_pred"] / 2)
    # fg_label = jax.nn.tanh(_max_merge(preds) / 2)
    fg_pred = jax.nn.sigmoid(preds["fg_pred"])
    fg_label = jax.lax.stop_gradient(jax.nn.sigmoid(_max_merge(preds)))

    # locs = jnp.floor(inputs["gt_locations"] + 0.5)
    # locs = locs.astype(int)
    # fg_label = fg_label.at[locs[:, 0], locs[:, 1]].set(1.0)

    assert fg_pred.shape == fg_label.shape

    # loss = (1.0 - fg_label) * fg_pred + fg_label * (1.0 - fg_pred)
    # p_t = fg_label * fg_pred + (1.0 -fg_label) * (1.0 - fg_pred)
    # loss = -jnp.log(jnp.clip(p_t, EPS, 1))
    loss = binary_focal_factor_loss(fg_pred, fg_label, gamma=1)
    loss = loss.mean()

    return loss


def aux_size_loss(batch, prediction, *, weight=0.01):
    """Auxillary loss to prevent model collapse"""
    preds = prediction
    inputs, labels, _ = unpack_x_y_sample_weight(batch)

    if labels is None:
        labels = {}

    if "gt_labels" in labels or "gt_masks" in labels:
        return 0.0

    height, width = inputs["image"].shape[:2]

    valid_locs = (
        (preds["instance_yc"] >= 0)
        & (preds["instance_yc"] < height)
        & (preds["instance_xc"] >= 0)
        & (preds["instance_xc"] < width)
    )

    areas = jnp.sum(
        preds["instance_output"], axis=(-1, -2), where=valid_locs, keepdims=True
    )
    areas = jnp.clip(areas, EPS, 1e6)
    loss = jax.lax.rsqrt(areas) * preds["instance_output"].shape[-1]

    mask = preds["instance_mask"]
    n_instances = jnp.count_nonzero(mask) + EPS
    loss = loss.sum(where=mask) / n_instances

    return loss * weight


def supervised_segmentation_loss(batch, prediction):
    preds = prediction
    _, labels, _ = unpack_x_y_sample_weight(batch)

    if "gt_labels" in labels:

        mask = (labels["gt_labels"] > 0).astype("float32")

    else:

        mask = labels["gt_image_mask"].astype("float32")

    return optax.sigmoid_binary_cross_entropy(preds["fg_pred"], mask).mean()


def collaborator_segm_loss(batch, prediction, *, sigma, pi):
    preds = prediction
    inputs, labels, _ = unpack_x_y_sample_weight(batch)

    if not "fg_pred" in preds:
        return 0.0

    if labels is None:
        labels = {}

    if "gt_image_mask" in labels or "gt_labels" in labels:
        return supervised_segmentation_loss(batch, prediction)
    else:
        return self_supervised_segmentation_loss(
            batch, prediction, offset_sigma=sigma, offset_scale=pi
        )


def collaborator_border_loss(batch, prediction):
    preds = prediction
    _, labels, _ = unpack_x_y_sample_weight(batch)

    if not "edge_pred" in preds:
        return 0.0

    if labels is None:
        labels = {}

    if "gt_labels" in labels or "gt_masks" in labels:
        return 0.0

    else:
        return self_supervised_edge_loss(batch, prediction)
