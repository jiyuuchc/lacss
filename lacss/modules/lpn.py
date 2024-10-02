from __future__ import annotations

from functools import partial
from typing import Sequence, Tuple, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from .common import DefaultUnpicklerMixin
from ..ops import non_max_suppression, distance_similarity
from ..typing import Array, ArrayLike

def generate_predictions(module, outputs):
    """
    Produce a list of proposal locations based on predication map, remove redundency with non_max_suppression
    """
    locations = outputs['pred_locs']
    scores = jax.nn.sigmoid(outputs['logits'])

    distance_threshold = module.nms_threshold
    output_size = module.max_output
    topk = module.pre_nms_topk
    score_threshold = module.min_score

    # sort and nms
    if topk <= 0 or topk > scores.size:
        topk = scores.size

    scores, indices = jax.lax.top_k(scores, topk)
    locations = locations[indices]

    threshold = 1 / distance_threshold / distance_threshold
    sel = non_max_suppression(
        scores,
        locations,
        output_size,
        threshold,
        score_threshold,
        return_selection=True,
    )

    idx_of_selected = jnp.argwhere(
        sel, size=output_size, fill_value=-1
    ).squeeze(-1)

    scores = jnp.where(idx_of_selected >= 0, scores[idx_of_selected], -1.0)
    locations = jnp.where(idx_of_selected[:, None] >= 0, locations[idx_of_selected], -1.0) 

    return dict(
        scores=scores,
        locations=locations,
    )


class LPN(nn.Module, DefaultUnpicklerMixin):
    """Location detection head

    Attributes:
        nms_threshold: non-max-supression threshold, if performing nms on detected locations.
        pre_nms_topk: max number of detections to be processed regardless of nms, ignored if negative
        max_output: number of detection outputs 
    """

    # network hyperparams
    feature_scale: int = 4
    dtype: Any = None

    # detection hyperparams
    nms_threshold: float = 8.0
    pre_nms_topk: int = -1
    max_output: int = 512
    min_score: float = 0.2

    @nn.compact
    def _block(self, feature: ArrayLike) -> dict:
        x = feature
        height, width = x.shape[:-1]

        logits = nn.Conv(1, (1, 1), dtype=self.dtype)(x)
        regressions = nn.Conv(2, (1, 1), dtype=self.dtype)(x)

        ref_locs = jnp.moveaxis(jnp.mgrid[:height, :width] + 0.5, 0, -1) * self.feature_scale
        locs = ref_locs + regressions * self.feature_scale

        return dict(
            regressions=regressions.reshape(-1, 2),
            logits = logits.reshape(-1),
            ref_locs=ref_locs.reshape(-1, 2),
            pred_locs=locs.reshape(-1, 2),
        )


    def __call__(self, feature: ArrayLike, mask: ArrayLike|None = None) -> dict:
        network_outputs = self._block(feature)
        predictions = generate_predictions(self, network_outputs)

        if mask is not None:
            pred_locs = tuple(jnp.floor(predictions["locations"]).transpose().astype(int))
            masked = mask[pred_locs]
            predictions['scores'] = jnp.where(masked, predictions['scores'], 0)
            predictions['locations'] = jnp.where(masked[:, None], predictions['locations'], -1)

        return dict(
            detector=network_outputs,
            predictions=predictions,
        )
