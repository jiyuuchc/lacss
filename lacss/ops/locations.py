from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from ..typing import *

Shape = Sequence[int]


def locations_to_labels(
    locations: ArrayLike, target_shape: Shape, threshold: float = 1.5
) -> tuple[Array, Array]:
    """Generate labels as LPN regression targets
    Args:
        locations: [N, 2] float32 true location values. scaled 0..1, masking out invalid with -1
        target_shape: (H, W)  int
        threshold: distance threshold for postive label
    Returns:
        score_target: [H, W, 1] int32
        regression_target: [H, W, 2] float tensor
    """
    height, width = target_shape
    threshold_sq = threshold * threshold

    flat_locations = locations * jnp.array(target_shape)

    mesh = jnp.mgrid[:height, :width] + 0.5  # [2, H, W]
    distances = flat_locations[:, :, None, None] - mesh  # [N, 2, H, W]
    distances_sq = (distances * distances).sum(axis=1)  # [N, H, W]

    # masking off invalid
    distances_sq = jnp.where(
        flat_locations[:, None, None, 0] >= 0, distances_sq, float("inf")
    )

    indices = jnp.argmin(distances_sq, axis=0, keepdims=True)  # [1, H, W]
    best_distances = jnp.take_along_axis(distances_sq, indices, 0)
    score_target = (best_distances < threshold_sq).astype(int)  # [1, H, W]
    score_target = score_target[0][:, :, None]  # [H, W, 1]

    indices = indices.repeat(2, axis=0)[None, ...]  # [1, 2, H, W]
    regression_target = jnp.take_along_axis(distances, indices, 0).squeeze(
        0
    )  # [2, H, W]
    regression_target = regression_target.transpose(1, 2, 0)  # [H, W, 2]

    return score_target, regression_target


def distance_similarity(pred_locations: ArrayLike, gt_locations: ArrayLike) -> Array:
    """Compute distance similarity matrix
    pairwise similarity = 1 / distance ^2
    Args:
      pred_locations: [N, 2] use -1 to mask out invalid locations
      gt_locations: [K, 2] use -1 to mask out invalid locations
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


def location_matching(
    pred_locations: ArrayLike, gt_locations: ArrayLike, threshold: float
) -> tuple[Array, Array]:
    """Match predicted location to gt locations
    Args:
      pred_locations:r [N, 2]
      gt_locations: [K, 2]
      threshold: float. Maximum distance to be matched
    Returns:
      matches: [N], indices of the matches location in gt list
      indicators: [N] bool
    """

    threshold = 1.0 / (threshold * threshold)

    distances_matrix = distance_similarity(pred_locations, gt_locations)

    matches = distances_matrix.argmax(axis=-1, keepdims=True)
    indicators = jnp.take_along_axis(distances_matrix, matches, -1) > threshold

    return matches.squeeze(-1), indicators.squeeze(-1)
