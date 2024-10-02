from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from xtrain import unpack_x_y_sample_weight

from ..typing import *

def get_image_shape(batch):
    inputs, _, _ = unpack_x_y_sample_weight(batch)
    img = inputs['image']
    if img.ndim == 3:
        return (1,) + img.shape[:2]
    else:
        return img.shape[:3]

def generalized_binary_loss(
    logits: ArrayLike, gt: ArrayLike, gamma: int | float, beta: int | float
) -> Array:
    pred = jax.nn.softmax(logits)

    p_t = pred * gt + (1 - pred) * (1 - gt)
    ff = 1 - p_t

    ce = optax.softmax_cross_entropy(logits, gt)

    loss = (ff**gamma) * (ce**beta)

    return loss


binary_focal_factor_loss = partial(generalized_binary_loss, beta=0)

binary_focal_crossentropy = partial(generalized_binary_loss, beta=1)

binary_crossentropy = partial(generalized_binary_loss, gamma=0, beta=1)


def sum_over_boolean_mask(loss: ArrayLike, mask: ArrayLike) -> Array:
    mask = mask.reshape(mask.shape[0], 1)

    loss = loss.reshape(loss.shape[0], -1)
    loss = loss.mean(axis=1, keepdims=True).sum(where=mask)

    return loss


def mean_over_boolean_mask(loss: ArrayLike, mask: ArrayLike) -> Array:
    mask = mask.reshape(mask.shape[0], 1)
    n_instances = jnp.count_nonzero(mask) + 1e-8

    loss = loss.reshape(loss.shape[0], -1)
    loss = loss.mean(axis=1, keepdims=True).sum(where=mask)
    loss /= n_instances

    return loss
