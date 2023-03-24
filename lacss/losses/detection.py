from functools import partial
from typing import Optional

import jax
import optax

from ..train.loss import Loss

jnp = jax.numpy

EPS = jnp.finfo("float32").eps


def _binary_focal_crossentropy(pred, gt, gamma=2.0):
    p_t = gt * pred + (1 - gt) * (1.0 - pred)
    focal_factor = (1.0 - p_t) ** gamma

    bce = -jnp.log(jnp.clip(p_t, EPS, 1.0))

    return focal_factor * bce


class DetectionLoss(Loss):
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, preds: dict, **kwargs) -> jnp.ndarray:
        def _inner(scores, gt_scores):
            score_loss = 0.0
            cnt = 1e-8
            for k in scores:
                score_loss += _binary_focal_crossentropy(
                    scores[k],
                    gt_scores[k],
                    self.gamma,
                ).sum()
                cnt += preds["lpn_scores"][k].size

            return score_loss / cnt

        return _inner(preds["lpn_scores"], preds["lpn_gt_scores"])


class LocalizationLoss(Loss):
    def __init__(self, delta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, preds: dict, **kwrags) -> jnp.ndarray:
        def _inner(regrs, gt_regrs, gt_scores):
            regr_loss = 0.0
            cnt = 1e-8
            for k in regrs:
                # h = optax.l2_loss(regrs[k], gt_regrs[k])
                h = optax.huber_loss(regrs[k], gt_regrs[k], delta=self.delta).mean(
                    axis=-1, keepdims=True
                )
                mask = gt_scores[k] > 0
                regr_loss += jnp.sum(h, where=mask)
                cnt += jnp.count_nonzero(mask)

            return regr_loss / (cnt + EPS)

        return _inner(
            preds["lpn_regressions"],
            preds["lpn_gt_regressions"],
            preds["lpn_gt_scores"],
        )


# class LocalizationLossAlt(Loss):
#     def __init__(self, delta=1.0, sigma=2.0, **kwargs):
#         super().__init__(**kwargs)
#         self.delta = delta
#         self.sigma_sq = sigma * sigma

#     def call(self, preds: dict, **kwargs) -> jnp.ndarray:
#         def _inner(regrs, gt_regrs, gt_scores):
#             regr_loss = 0.0
#             cnt = 1e-8
#             for k in regrs:
#                 # h = optax.l2_loss(regrs[k], gt_regrs[k])
#                 h = optax.huber_loss(regrs[k], gt_regrs[k], delta=self.delta).mean(
#                     axis=-1
#                 )
#                 w = jnp.exp(-(gt_regrs[k] ** 2).sum(axis=-1) / self.sigma_sq)
#                 regr_loss += jnp.sum(h * w)
#                 cnt += w.sum()
#             return regr_loss / cnt

#         return jax.vmap(_inner)(
#             preds["lpn_regressions"],
#             preds["lpn_gt_regressions"],
#             preds["lpn_gt_scores"],
#         )


class LPNLoss(Loss):
    def __init__(self, delta=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.det_loss = DetectionLoss(gamma=gamma)
        self.loc_loss = LocalizationLoss(delta=delta)

    def call(self, preds: dict, **kwargs):
        return self.det_loss.call(preds=preds) + self.loc_loss(preds=preds)
