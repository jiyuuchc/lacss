from __future__ import annotations

from functools import partial
from typing import Sequence, Tuple, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from .lpn import generate_predictions
from .common import DefaultUnpicklerMixin
from ..typing import Array, ArrayLike, Any

class LPN3D(nn.Module, DefaultUnpicklerMixin):
    """Location 3d detection head

    Attributes:
        n_layers: num of conv layers for feature mixing
        nms_threshold: non-max-supression threshold, if performing nms on detected locations.
        pre_nms_topk: max number of detections to be processed regardless of nms, ignored if negative
        max_output: number of detection outputs 
    """

    # network hyperparams
    n_layers: int = 2
    dim: int = 384
    feature_scale: int = 4
    dtype: Any = None

    # detection hyperparams
    nms_threshold: float = 8.0
    pre_nms_topk: int = -1
    max_output: int = 256
    min_score: float = 0.2

    def _block(self, feature: Array) -> dict:
        x = feature

        depth, height, width = x.shape[:-1]

        logits = nn.Conv(1, (1, 1), dtype=self.dtype)(x)
        regressions = nn.Conv(3, (1, 1), dtype=self.dtype)(x)

        ref_locs = jnp.moveaxis(jnp.mgrid[:depth, :height, :width] + 0.5, 0, -1) * self.feature_scale
        locs = ref_locs + regressions * self.feature_scale

        return dict(
            regressions=regressions.reshape(-1, 3),
            logits = logits.reshape(-1),
            ref_locs=ref_locs.reshape(-1, 3),
            pred_locs=locs.reshape(-1, 3),
        )


    def _mix_feaures(self, feature: Array) -> Array:
        x = feature
        dim = self.dim
        for _ in range(self.n_layers):
            x = nn.Conv(dim, (3,3,3), dtype=self.dtype)(x)
            x = nn.gelu(x)

        return x

    @nn.compact
    def __call__(self, feature: ArrayLike, mask: ArrayLike|None = None) -> dict:
        feature = jnp.asarray(feature)

        x = self._mix_feaures(feature)
        x = self._block(x)
        predictions = generate_predictions(self, x)

        if mask is not None:
            pred_locs = jnp.floor(predictions["locations"]).astype(int)
            pred_locs_ = jnp.clip(pred_locs, 0, jnp.array(mask.shape) - 1)
            masked = jnp.where(
                (pred_locs == pred_locs_).all(axis=1), 
                mask[tuple(pred_locs_.transpose())], 
                False
            )
            predictions['scores'] = jnp.where(masked, predictions['scores'], 0)
            # predictions['locations'] = jnp.where(masked[:, None], predictions['locations'], -1)

        return dict(
            detector=x,
            predictions=predictions,
        )
