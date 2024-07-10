from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from ..typing import *


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


def location_matching(
    pred_locations: ArrayLike, gt_locations: ArrayLike, threshold: float
) -> tuple[Array, Array]:
    """Match predicted location to gt locations

    Args:
      pred_locations:r [N, d]
      gt_locations: [K, d]
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
