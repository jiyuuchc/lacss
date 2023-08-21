from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import optax

from .common import binary_focal_crossentropy

jnp = jax.numpy

EPS = jnp.finfo("float32").eps

# def _binary_focal_crossentropy(pred, gt, gamma=2.0):
#     p_t = gt * pred + (1 - gt) * (1.0 - pred)
#     focal_factor = (1.0 - p_t) ** gamma

#     bce = -jnp.log(jnp.clip(p_t, EPS, 1.0))

#     return focal_factor * bce


def detection_loss(batch, prediction, *, gamma=2.0):
    """LPN detection loss"""

    preds = prediction

    scores = preds["lpn_scores"]
    gt_scores = preds["lpn_gt_scores"]

    score_loss = 0.0
    cnt = EPS
    for k in scores:
        score_loss += binary_focal_crossentropy(
            scores[k],
            gt_scores[k],
            gamma=gamma,
        ).sum()
        cnt += preds["lpn_scores"][k].size

    return score_loss / cnt


def localization_loss(batch, prediction, *, delta=1.0):
    """LPN localization loss"""
    preds = prediction

    regrs = preds["lpn_regressions"]
    gt_regrs = preds["lpn_gt_regressions"]
    gt_scores = preds["lpn_gt_scores"]

    regr_loss = 0.0
    cnt = 1e-8
    for k in regrs:
        # h = optax.l2_loss(regrs[k], gt_regrs[k])
        h = optax.huber_loss(regrs[k], gt_regrs[k], delta=delta).mean(
            axis=-1, keepdims=True
        )
        mask = gt_scores[k] > 0
        regr_loss += jnp.sum(h, where=mask)
        # cnt += jnp.count_nonzero(mask)
        cnt += gt_scores[k].size

    return regr_loss / (cnt + EPS)


def lpn_loss(batch, prediction, *, gamma=2.0, w1=1.0, w2=1.0):
    """LPN loss"""

    return (
        detection_loss(batch, prediction, gamma=gamma) * w1
        + localization_loss(batch, prediction) * w2
    )
