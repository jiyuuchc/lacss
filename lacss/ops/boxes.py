""" Ops on bounding-boxes

All functions here are degisned to work as either a numpy op or a jax op depending on
the data type of the input.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..typing import *


def box_area(box: ArrayLike) -> ArrayLike:
    """Computes area of boxes.
    Args:
      box: a float Tensor with [..., N, 2d].

    Returns:
      a float Tensor with [..., N]
    """
    ndim = box.shape[-1] // 2
    assert ndim * 2 == box.shape[-1]

    min_vals = box[..., :ndim]
    max_vals = box[..., ndim:]

    return (max_vals - min_vals).prod(axis=-1)


def box_intersection(gt_boxes: ArrayLike, boxes: ArrayLike) -> ArrayLike:
    """Compute pairwise intersection areas between boxes.

    Args:
      gt_boxes: [..., N, 2d]
      boxes: [..., M, 2d]

    Returns:
      a float Tensor with shape [..., N, M] representing pairwise intersections.
    """
    if isinstance(gt_boxes, jnp.ndarray) & isinstance(boxes, jnp.ndarray):
        minimum = jnp.minimum
        maximum = jnp.maximum
    else:
        minimum = np.minimum
        maximum = np.maximum

    ndim = gt_boxes.shape[-1] // 2
    assert ndim * 2 == gt_boxes.shape[-1]
    assert ndim * 2 == boxes.shape[-1]

    min_vals_1 = gt_boxes[..., None, :ndim] # [..., N, 1, d]
    max_vals_1 = gt_boxes[..., None, ndim:]
    min_vals_2 = boxes[..., None, :, :ndim] # [..., 1, M, d]
    max_vals_2 = boxes[..., None, :, ndim:]

    min_max = minimum(max_vals_1, max_vals_2) #[..., N, M, d]
    max_min = maximum(min_vals_1, min_vals_2)

    intersects = maximum(0, min_max - max_min) # [..., N, M, d]

    return intersects.prod(axis=-1)


def box_iou_similarity(gt_boxes: ArrayLike, boxes: ArrayLike) -> ArrayLike:
    """Computes pairwise intersection-over-union between box collections.

    Args:
      gt_boxes: a float Tensor with [..., N, 2d].
      boxes: a float Tensor with [..., M, 2d].

    Returns:
      a Tensor with shape [..., N, M] representing pairwise iou scores.
    """
    intersections = box_intersection(gt_boxes, boxes)
    gt_boxes_areas = box_area(gt_boxes)
    boxes_areas = box_area(boxes)
    unions = gt_boxes_areas[..., None] + boxes_areas[..., None, :] - intersections

    ious = intersections / (unions + 1e-6)

    return ious

def yxhw_iou_similarity(yxhw_a, yxhw_b):
    """Computes pairwise IOU on bbox in yxhw format

    Args:
      gt_boxes: a float Tensor with [..., N, 2d].
      boxes: a float Tensor with [..., M, 2d].

    Returns:
      a Tensor with shape [..., N, M] representing pairwise iou scores.
    """
    def _yxhw2box(yxhw):
        dim = yxhw.shape[-1] // 2        
        center, span = yxhw[..., :dim], yxhw[..., dim:]
        return jnp.c_[center - span / 2, center + span / 2]
      
    return box_iou_similarity(_yxhw2box(yxhw_a), _yxhw2box(yxhw_b))

def iou_loss(gt_boxes: ArrayLike, boxes: ArrayLike) -> ArrayLike:
    """ IOU loss = 1 - IOU
    Args:
      gt_boxes: a float Tensor with [..., N, 2d].
      boxes: a float Tensor with [..., N, 2d].

    Returns:
      a Tensor with shape [..., N] representing pairwise loss.
    """

    loss = 1 - jax.vmap(box_iou_similarity, in_axes=-2, out_axes=-2)(
      gt_boxes[..., None, :, :], boxes[..., None, :, :],
    )
    return loss.squeeze(axis=-3)


def generalized_iou_loss(gt_boxes: ArrayLike, boxes: ArrayLike) -> ArrayLike:
    """ Loss_GIoU = 1 - IoU + |C - B union B_GT| / |C|
    where C is the smallest enclosing box for both B and B_GT. The resulting value has a gradient 
    for non-overlapping boxes. See  Zheng et al. [AAAI 2020] 

    Args:
      gt_boxes: a float Tensor with [..., N, 2d].
      boxes: a float Tensor with [..., N, 2d].

    Returns:
      a Tensor with shape [..., N] representing pairwise loss.
    """
    if isinstance(gt_boxes, jnp.ndarray) & isinstance(boxes, jnp.ndarray):
        minimum = jnp.minimum
        maximum = jnp.maximum
    else:
        minimum = np.minimum
        maximum = np.maximum

    ndim = gt_boxes.shape[-1] // 2
    assert ndim * 2 == gt_boxes.shape[-1]
    assert ndim * 2 == boxes.shape[-1]

    intersections = jax.vmap(box_intersection, in_axes=-2, out_axes=-2)(
      gt_boxes[..., None, :, :], boxes[..., None, :, :],
    ).squeeze(-3)
    gt_boxes_areas = box_area(gt_boxes)
    boxes_areas = box_area(boxes)
    unions = gt_boxes_areas + boxes_areas - intersections

    ious = intersections / (unions + 1e-6)

    mins = minimum(gt_boxes[..., :ndim], boxes[..., :ndim])
    maxs = maximum(gt_boxes[..., ndim:], boxes[..., ndim:])
    intersects = maximum(0, maxs - mins) # [..., N, d]

    C = intersects.prod(axis=-1)
    
    return 1 - ious + (C - unions) / (C + 1e-6)

def distance_iou_loss(gt_boxes: ArrayLike, boxes: ArrayLike) -> ArrayLike:
    """ Loss_distance_iou = 1 - IoU + \rho2(B, B_GT) / c2
    The correction term is "distance_sq between box center" / "distance_sq of enclosing box cornor"

    Args:
      gt_boxes: a float Tensor with [..., N, 2d].
      boxes: a float Tensor with [..., N, 2d].

    Returns:
      a Tensor with shape [..., N] representing pairwise loss.    
    """
    if isinstance(gt_boxes, jnp.ndarray) & isinstance(boxes, jnp.ndarray):
        minimum = jnp.minimum
        maximum = jnp.maximum
    else:
        minimum = np.minimum
        maximum = np.maximum

    ndim = gt_boxes.shape[-1] // 2
    assert ndim * 2 == gt_boxes.shape[-1]
    assert ndim * 2 == boxes.shape[-1]

    ious = jax.vmap(box_iou_similarity, in_axes=-2, out_axes=-2)(
      gt_boxes[..., None, :, :], boxes[..., None, :, :],
    ).squeeze(-3)
    mins = minimum(gt_boxes[..., :ndim], boxes[..., :ndim])
    maxs = maximum(gt_boxes[..., ndim:], boxes[..., ndim:])
    intersects = maximum(0, maxs - mins) # [..., N, d]
    c2 = (intersects ** 2).sum(axis=-1)

    center_gt = (gt_boxes[..., :ndim] + gt_boxes[..., ndim:]) / 2
    center = (boxes[..., :ndim] + boxes[..., ndim:]) / 2
    rho2 = ((center_gt - center) ** 2).sum(axis=-1)

    return 1 - ious + rho2/c2


def distance_similarity(pred_locations: ArrayLike, gt_locations: ArrayLike) -> Array:
    """Compute distance similarity matrix

        pairwise similarity = 1 / distance ^2

    Args:
        pred_locations: [N, d] use -1 to mask out invalid locations
        gt_locations: [K, d] use -1 to mask out invalid locations

    Returns:
        similarity_matrix: [N, k]
    """

    distances_matrix = pred_locations[:, None, :] - gt_locations
    distances_matrix = jnp.square(distances_matrix).sum(axis=-1)

    sm = 1.0 / (distances_matrix + 1e-8)

    # negative locations are invalid
    sm = jnp.where((pred_locations < 0).all(axis=-1)[:, None], 0, sm)
    sm = jnp.where((gt_locations < 0).all(axis=-1), 0, sm)

    return sm


def feature_matching(
    features_a: ArrayLike, features_b: ArrayLike, threshold: float,
    *,
    similarity_fn = None,
) -> tuple[Array, Array]:
    """Match predicted location to gt locations

    Args:
      features_a:r [N, d], points or bboxes
      features_b: [K, d], points or bboxes
      threshold: float. Min similarity score for match

    Keyword Args:
      similarity_fn: optional custom function for computing similarity score
          fn(feautes_a, feature_b) -> similarity matrix

    Returns:
      matches: [N], indices of the matches row in b 
      indicators: [N] bool
    """
    dim = features_a.shape[-1]
    if similarity_fn is None:
        similarity_fn = distance_similarity if dim <=3 else yxhw_iou_similarity

    sm = similarity_fn(features_a, features_b)
    matches = sm.argmax(axis=-1, keepdims=True)

    indicators = jnp.take_along_axis(sm, matches, -1) > threshold

    return matches.squeeze(-1), indicators.squeeze(-1)


def match_and_replace(gt_features, pred_features, threshold, *, similarity_fn = None):
    """replacing gt_locations with pred_locations if the close enough
    1. Each pred_location is matched to the closest gt_location
    2. For each gt_location, pick the matched pred_location with highest score
    3. if the picked pred_location is within threshold distance, replace the gt_location with the pred_location
    """

    n_gt_locs = gt_features.shape[0]
    n_pred_locs = pred_features.shape[0]

    matched_id, indicators = feature_matching(
        pred_features, 
        gt_features, 
        threshold,
        similarity_fn = similarity_fn 
    )
    matched_id = jnp.where(indicators, matched_id, -1)

    matching_matrix = (
        matched_id[None, :] == jnp.arange(n_gt_locs)[:, None]
    )  # true at matched gt(row)/pred(col), at most one true per col
    last_col = jnp.ones([n_gt_locs, 1], dtype=bool)
    matching_matrix = jnp.concatenate(
        [matching_matrix, last_col], axis=-1
    )  # last col is true

    # first true of every row
    matched_loc_ids = jnp.argmax(matching_matrix, axis=-1)

    training_locations = jnp.where(
        matched_loc_ids[:, None] == n_pred_locs,  # true: failed match
        gt_features,
        pred_features[matched_loc_ids] # TODO out-of-bound error on cpu
    )

    return training_locations
