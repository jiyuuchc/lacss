from __future__ import annotations

from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..ops import location_matching, sorted_non_max_suppression
from ..typing import *


class Detector(nn.Module):
    """A weightless module that conver LPN output to a list of locations. Non-max-supression is
        appplied to remove redundant detections.

    Attributes:

        train_nms_threshold: Threshold for non-max-suppression during training
        train_pre_nms_topk: If > 0, only the top_k scored locations will be analyzed during training
        train_max_output: Maximum number of outputs during training
        train_min_score: Mininum scores to be considered during training
        max_proposal_offset: During training, if a detected location is with in this distance threshold
            of a ground-truth location, replacing the ground truth location with the detection one for
            segmentation purpose.
        test_nms_threshold: Threshold for non-max-suppression during testing
        test_pre_nms_topk: If > 0, only the top_k scored locations will be analyzed during testing
        test_max_output: Maximum number of outputs during testing
        test_min_score: Mininum scores to be considered during testing

    """

    train_nms_threshold: float = 8.0
    train_pre_nms_topk: int = -1
    train_max_output: int = 2560
    train_min_score: float = 0.2
    max_proposal_offset: float = 12
    test_nms_threshold: float = 8.0
    test_pre_nms_topk: int = -1
    test_max_output: int = 2560
    test_min_score: float = 0.2
    # training: bool = True

    def _proposal_locations(self, lpn_scores, lpn_regression, training: bool = None):
        """
        Produce a list of proposal locations based on predication map, remove redundency with non_max_suppression
        Args:
            lpn_score: dict {'lvl': [height, width, 1]} lpn score prediction
            lpn_regression: dict {'lvl': [height, width, 2]} lpn regression prediction
        Returns:
            scores: [N], sorted scores
            locations: [N, 2], proposed location
        """
        if training:
            distance_threshold = self.train_nms_threshold
            output_size = self.train_max_output
            topk = self.train_pre_nms_topk
            score_threshold = self.train_min_score
        else:
            distance_threshold = self.test_nms_threshold
            output_size = self.test_max_output
            topk = self.test_pre_nms_topk
            score_threshold = self.test_min_score

        # preprocess
        scores = []
        locations = []
        for k in lpn_scores:
            s = lpn_scores[k].squeeze(-1)
            r = lpn_regression[k]
            height, width = s.shape
            loc = jnp.mgrid[:height, :width] + 0.5
            loc = loc.transpose(1, 2, 0) + r

            # set invalid location score to 0
            is_valid = (
                (loc > 0.0).all(axis=-1)
                & (loc[:, :, 0] < height)
                & (loc[:, :, 1] < width)
            )
            s = jnp.where(is_valid, s, 0)

            loc *= 2 ** int(k)  # scale to origianl image size

            locations.append(loc.reshape(-1, 2))
            scores.append(s.reshape(-1))

        scores = jnp.concatenate(scores, axis=0)
        locations = jnp.concatenate(locations, axis=0)

        # select topk
        if topk <= 0 or topk > scores.size:
            topk = scores.size

        # refine
        if distance_threshold > 0:  # nms

            scores, selections = jax.lax.top_k(scores, topk)
            locations = locations[selections]

            threshold = 1 / distance_threshold / distance_threshold
            scores, locations = sorted_non_max_suppression(
                scores,
                locations,
                output_size,
                threshold,
                score_threshold,
            )

        else:

            sel = jnp.where(scores >= score_threshold, size=output_size, fill_value=-1)
            sel = sel[0]
            scores = jnp.where(sel >= 0, scores[sel], -1.0)
            locations = jnp.where((sel >= 0)[:, None], locations[sel], -1.0)

        return scores, locations

    def _gen_train_locations(self, gt_locations, pred_locations):
        """replacing gt_locations with pred_locations if the close enough
        gt_locations: [N, 2] tensor
        pred_locations: [M, 2] tensor, sorted with scores

        1. Each pred_location is matched to the closest gt_location
        2. For each gt_location, pick the matched pred_location with highest score
        3. if the picked pred_location is within threshold distance, replace the gt_location with the pred_location
        """

        threshold = self.max_proposal_offset

        n_gt_locs = gt_locations.shape[0]
        n_pred_locs = pred_locations.shape[0]

        matched_id, indicators = location_matching(
            pred_locations, gt_locations, threshold
        )
        matched_id = jnp.where(indicators, matched_id, -1)

        matching_matrix = (
            matched_id[None, :] == jnp.arange(n_gt_locs)[:, None]
        )  # true at matched gt(row)/pred(col), at most one true per col
        last_col = jnp.ones([n_gt_locs, 1], dtype=bool)
        matching_matrix = jnp.concatenate(
            [matching_matrix, last_col], axis=-1
        )  # last col is true

        # first true of every row
        matched_loc_ids = jnp.argmax(matching_matrix, axis=-1)

        training_locations = jnp.where(
            matched_loc_ids[:, None] == n_pred_locs,  # true: failed match
            gt_locations,
            pred_locations[
                matched_loc_ids, :
            ],  # out-of-bound error silently dropped in jax
        )

        return training_locations

    def __call__(
        self,
        scores: Mapping[str, ArrayLike],
        regressions: Mapping[str, ArrayLike],
        gt_locations: Optional[ArrayLike] = None,
        *,
        training: Optional[bool] = None,
    ) -> DataDict:
        """
        Args:
                scores: {'scale', [H, W, 1]) output from LPN
                regressions: {'scale': [H, W, 2]} output from LPN
                gt_locations: Ground-truth call locations. This can be an array  of [N, 2] (training) or
                    None (inference). Maybe padded with -1
                training: Whether to run the module in training mode or not

        Returns:
                a dictionary of values.

                    * pred_locations: Sorted array based on scores
                    * pred_scores: Sorted array, padded with -1
                    * training_locations: This value exists during training only.

        """
        scores = jax.lax.stop_gradient(scores)
        regressions = jax.lax.stop_gradient(regressions)

        if gt_locations is not None:

            if self.max_proposal_offset <= 0:
                return dict(
                    training_locations=gt_locations,
                )

            proposed_scores, proposed_locations = self._proposal_locations(
                scores, regressions, True
            )

            outputs = dict(
                pred_locations=proposed_locations,
                pred_scores=proposed_scores,
                training_locations=self._gen_train_locations(
                    gt_locations,
                    proposed_locations,
                ),
            )

        else:

            proposed_scores, proposed_locations = self._proposal_locations(
                scores, regressions, False
            )
            outputs = dict(
                pred_locations=proposed_locations,
                pred_scores=proposed_scores,
            )

        return outputs
