""" Ops on bounding-boxes

All functions here are degisned to work as either a numpy op or a jax op depending on
the data type of the input.
"""
from __future__ import annotations

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

def iou_loss(gt_boxes: ArrayLike, boxes: ArrayLike) -> ArrayLike:
    """ IOU loss = 1 - IOU
    Args:
      gt_boxes: a float Tensor with [..., N, 2d].
      boxes: a float Tensor with [..., M, 2d].

    Returns:
      a Tensor with shape [..., N, M] representing pairwise loss.
    """
    return 1 - box_iou_similarity(gt_boxes, boxes)


def generalized_iou_loss(gt_boxes: ArrayLike, boxes: ArrayLike) -> ArrayLike:
    """ Loss_GIoU = 1 - IoU + |C - B union B_GT| / |C|
    where C is the smallest enclosing box for both B and B_GT. The resulting value has a gradient 
    for non-overlapping boxes. See  Zheng et al. [AAAI 2020] 

    Args:
      gt_boxes: a float Tensor with [..., N, 2d].
      boxes: a float Tensor with [..., M, 2d].

    Returns:
      a Tensor with shape [..., N, M] representing pairwise loss.
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

    intersections = box_intersection(gt_boxes, boxes)
    gt_boxes_areas = box_area(gt_boxes)
    boxes_areas = box_area(boxes)
    unions = gt_boxes_areas[..., None] + boxes_areas[..., None, :] - intersections

    ious = intersections / (unions + 1e-6)

    mins = minimum(gt_boxes[..., None, :ndim], boxes[..., None, :, :ndim])
    maxs = maximum(gt_boxes[..., None, ndim:], boxes[..., None, :, ndim:])
    intersects = maximum(0, maxs - mins) # [..., N, M, d]

    C = intersects.prod(axis=-1)
    
    return 1 - ious + (C - unions) / (C + 1e-6)

def distance_iou_loss(gt_boxes: ArrayLike, boxes: ArrayLike) -> ArrayLike:
    """ Loss_distance_iou = 1 - IoU + \rho2(B, B_GT) / c2
    The correction term is "distance_sq between box center" / "distance_sq of enclosing box cornor"

    Args:
      gt_boxes: a float Tensor with [..., N, 2d].
      boxes: a float Tensor with [..., M, 2d].

    Returns:
      a Tensor with shape [..., N, M] representing pairwise loss.    
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

    ious = box_iou_similarity(gt_boxes, boxes)
    mins = minimum(gt_boxes[..., None, :ndim], boxes[..., None, :, :ndim])
    maxs = maximum(gt_boxes[..., None, ndim:], boxes[..., None, :, ndim:])
    intersects = maximum(0, maxs - mins) # [..., N, M, d]
    c2 = (intersects ** 2).sum(axis=-1)

    center_gt = (gt_boxes[..., :ndim] + gt_boxes[..., ndim:]) / 2
    center = (boxes[..., :ndim] + boxes[..., ndim:]) / 2
    rho2 = ((center_gt[..., None, :] - center[..., None, :, :]) ** 2).sum(axis=-1)

    return 1 - ious + rho2/c2
