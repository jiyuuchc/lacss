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
      box: a float Tensor with [..., N, 4].

    Returns:
      a float Tensor with [..., N]
    """
    if isinstance(box, jnp.ndarray):
        split = jnp.split
    else:
        split = np.split

    y_min, x_min, y_max, x_max = split(box, 4, axis=-1)
    return ((y_max - y_min) * (x_max - x_min)).squeeze(axis=-1)


def box_intersection(gt_boxes: ArrayLike, boxes: ArrayLike) -> ArrayLike:
    """Compute pairwise intersection areas between boxes.

    Args:
      gt_boxes: [..., N, 4]
      boxes: [..., M, 4]

    Returns:
      a float Tensor with shape [..., N, M] representing pairwise intersections.
    """
    if isinstance(gt_boxes, jnp.ndarray) & isinstance(boxes, jnp.ndarray):
        split = jnp.split
        minimum = jnp.minimum
        maximum = jnp.maximum
    else:
        split = np.split
        minimum = np.minimum
        maximum = np.maximum

    y_min1, x_min1, y_max1, x_max1 = split(gt_boxes, 4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = split(boxes, 4, axis=-1)

    # [N, M] or [B, N, M]
    y_min_max = minimum(y_max1, y_max2.swapaxes(-1, -2))
    y_max_min = maximum(y_min1, y_min2.swapaxes(-1, -2))
    x_min_max = minimum(x_max1, x_max2.swapaxes(-1, -2))
    x_max_min = maximum(x_min1, x_min2.swapaxes(-1, -2))

    intersect_heights = y_min_max - y_max_min
    intersect_widths = x_min_max - x_max_min
    intersect_heights = maximum(0, intersect_heights)
    intersect_widths = maximum(0, intersect_widths)

    return intersect_heights * intersect_widths


def box_iou_similarity(gt_boxes: ArrayLike, boxes: ArrayLike) -> ArrayLike:
    """Computes pairwise intersection-over-union between box collections.

    Args:
      gt_boxes: a float Tensor with [..., N, 4].
      boxes: a float Tensor with [..., M, 4].

    Returns:
      a Tensor with shape [..., N, M] representing pairwise iou scores.
    """
    intersections = box_intersection(gt_boxes, boxes)
    gt_boxes_areas = box_area(gt_boxes)
    boxes_areas = box_area(boxes)
    unions = gt_boxes_areas[..., None] + boxes_areas[..., None, :] - intersections

    ious = intersections / (unions + 1e-6)

    return ious
