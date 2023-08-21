from __future__ import annotations

from functools import partial

import jax.numpy as jnp

from ..typing import *


def generalized_binary_loss(
    pred: ArrayLike, gt: ArrayLike, gamma: int | float, beta: int | float
) -> Array:
    EPS = jnp.finfo(jnp.promote_types(pred, gt)).eps

    p_t = pred * gt + (1 - pred) * (1 - gt)

    ff = 1 - p_t
    ce = -jnp.log(jnp.clip(p_t, EPS, 1))

    loss = (ff**gamma) * (ce**beta)

    return loss


binary_focal_factor_loss = partial(generalized_binary_loss, beta=0)

binary_focal_crossentropy = partial(generalized_binary_loss, beta=1)

binary_crossentropy = partial(generalized_binary_loss, gamma=0, beta=1)


def mean_over_boolean_mask(loss: ArrayLike, mask: ArrayLike) -> Array:
    mask = mask.reshape(mask.shape[0], 1)
    n_instances = jnp.count_nonzero(mask) + 1e-8

    loss = loss.reshape(loss.shape[0], -1)
    loss = loss.mean(axis=1, keepdims=True).sum(where=mask)
    loss /= n_instances

    return loss
