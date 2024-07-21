from __future__ import annotations

from typing import Sequence, Tuple, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from ..losses.common import binary_focal_crossentropy
from ..ops import non_max_suppression, location_matching
from ..typing import Array, ArrayLike

EPS = jnp.finfo("float32").eps


class LPN(nn.Module):
    """Location detection head

    Attributes:
        feature_levels: Input feature level, e.g. [0, 1]
        feature_level_scales: The scaling of each feature level. 
        conv_spec: Conv layer specification
        detection_n_cls: number of classes
        nms_threshold: non-max-supression threshold, if performing nms on detected locations.
        pre_nms_topk: max number of detections to be processed regardless of nms, ignored if negative
        max_output: number of detection outputs 

        detection_roi: Parameter for label smoothing for focal loss
        detection_loss_gamma: gamma value for detection focal loss
        detection_loss_delta: delta value for offset regression huber loss
    """

    # network hyperparams
    feature_levels: Sequence[int] = (0, 1, 2)
    feature_level_scales: Sequence[int] = (4, 8, 16)
    conv_spec: Sequence[int] = (192, 192, 192, 192)
    normalization: Callable[[None], nn.Module]=nn.GroupNorm
    activation: Callable[[Array], Array] = nn.relu

    detection_n_cls: int = 1

    # loss hyperparams
    detection_roi: float = 8.0
    detection_loss_gamma: float = 2.0
    detection_loss_delta: float = 1.0

    # detection hyperparams
    nms_threshold: float = 8.0
    pre_nms_topk: int = -1
    max_output: int = 512
    min_score: float = 0.2
    max_proposal_offset: float = 12.0
    z_scale:float = 5.0

    def _best_match(self, gt_locations, pred_locations):
        """replacing gt_locations with pred_locations if the close enough
        1. Each pred_location is matched to the closest gt_location
        2. For each gt_location, pick the matched pred_location with highest score
        3. if the picked pred_location is within threshold distance, replace the gt_location with the pred_location
        """

        threshold = self.max_proposal_offset

        n_gt_locs = gt_locations.shape[0]
        n_pred_locs = pred_locations.shape[0]

        matched_id, indicators = location_matching(
            pred_locations * jnp.asarray([self.z_scale, 1, 1]), 
            gt_locations * jnp.asarray([self.z_scale, 1, 1]), 
            threshold,
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

    def _to_targets(
        self,
        locations: ArrayLike,
        clses: ArrayLike,
        feature_shape: tuple[int, int],
        scale: float,
    ) -> tuple[Array, Array]:
        """Generate labels as LPN regression targets

        Returns:
            is_target: [H, W, n_cls] int32
            regression_target: [H, W, 2] float tensor
        """
        depth, height, width = feature_shape

        threshold_sq = self.detection_roi ** 2

        flat_locations = jnp.expand_dims(
            locations * jnp.asarray([self.z_scale, 1, 1]),
            (1, 2, 3),
        ) #[N, 1, 1, 1, 3]

        mesh = (
            jnp.moveaxis(jnp.mgrid[:depth, :height, :width] + 0.5, 0, -1) 
            * jnp.asarray([self.z_scale, scale, scale])
        )  # [D, H, W, 3]
        distances = flat_locations - mesh  # [N, D, H, W, 3]
        distances_sq = (distances * distances).sum(axis=-1)  # [N, D, H, W]

        # masking off invalid
        distances_sq = jnp.where(
            (flat_locations >= 0).all(axis=-1), distances_sq, float("inf")
        )

        indices = jnp.argmin(
            distances_sq, axis=0, keepdims=True
        )  # [1, D, H, W] map to nearest gt-loc
        cls_map = clses[
            indices.squeeze(0)
        ]  # [D, H, W] map to the cls of the nearest gt-loc

        best_distances = jnp.take_along_axis(distances_sq, indices, 0).squeeze(0)
        is_target = (
            best_distances < threshold_sq
        )  # [D, H, W] indicate whether the pixel is close to a gt-loc
        cls_target = jnp.where(is_target, cls_map, self.detection_n_cls)  # [D, H, W]

        indices = indices[...,None].repeat(3, axis=-1)  # [1, D, H, W, 3]
        regression_target = jnp.take_along_axis(distances, indices, 0)
        regression_target = regression_target.squeeze(0) # [D, H, W, 3]
        regression_target = regression_target / jnp.asarray([self.z_scale, scale, scale]) # normalize

        return cls_target, regression_target

    def _compute_losses(self, network_outputs, gt_locations, gt_cls):
        cls_loss, regr_loss = [], []
        n_targets = 0

        for block_out in network_outputs.values():
            depth, height, width, _ = block_out["cls_logits"].shape

            if gt_cls is None:
                gt_cls = jnp.zeros(gt_locations.shape[0], dtype=int)

            cls_target, regression_target = self._to_targets(
                gt_locations,
                gt_cls,
                (depth, height, width),
                block_out["scale"],
            )

            self.sow("intermediates", "cls_target", cls_target)
            self.sow("intermediates", "regression_target", regression_target)

            cls_loss.append(
                binary_focal_crossentropy(
                    jax.nn.softmax(block_out["cls_logits"]),
                    jax.nn.one_hot(cls_target, self.detection_n_cls+1),
                    gamma=self.detection_loss_gamma,
                ).mean(axis=-1).reshape(-1)
            )

            regr_loss.append(
                jnp.where(
                    cls_target != self.detection_n_cls,
                    optax.huber_loss(
                        block_out["regressions"],
                        regression_target,
                        delta=self.detection_loss_delta,
                    ).mean(axis=-1),
                    0,
                ).reshape(-1)
            )

            n_targets += jnp.count_nonzero(cls_target != self.detection_n_cls)

        cls_loss = jnp.concatenate(cls_loss).mean()
        regr_loss = jnp.concatenate(regr_loss).sum() / (n_targets + 1e-8)

        return dict(
            lpn_localization_loss=regr_loss,
            lpn_detection_loss=cls_loss,
        )

    def _generate_predictions(self, network_outputs):
        """
        Produce a list of proposal locations based on predication map, remove redundency with non_max_suppression
        """
        distance_threshold = self.nms_threshold
        output_size = self.max_output
        topk = self.pre_nms_topk
        score_threshold = self.min_score

        network_outputs = jax.lax.stop_gradient(network_outputs)

        # preprocess
        scores, locations, clss = [], [], []
        for block_out in network_outputs.values():
            logits = jax.nn.softmax(block_out["cls_logits"])
            scores_ = logits[..., :-1].sum(axis=-1) # [D, H, W]
            clss_ = logits[..., :-1].argmax(axis=-1)

            depth, height, width = scores_.shape
            loc_ = jnp.moveaxis(jnp.mgrid[:depth, :height, :width] + 0.5, 0, -1)  # [D, H, W, 3]
            loc_ = loc_ + block_out["regressions"]

            # set invalid location score to -1
            max_values = jnp.asarray([depth, height, width])
            is_valid = (loc_ > 0.0).all(axis=-1) & (loc_ < max_values).all(axis=-1)
            scores_ = jnp.where(is_valid, scores_, -1)

            loc_ = loc_ * jnp.asarray([self.z_scale, block_out["scale"], block_out["scale"]]) # back to physical scale

            scores.append(scores_.reshape(-1))
            locations.append(loc_.reshape(-1, 3))
            clss.append(clss_.reshape(-1))

        scores, locations, clss = map(jnp.concatenate, (scores, locations, clss))

        assert scores.shape[0] == locations.shape[0]
        assert scores.shape[0] == clss.shape[0]

        # sort and nms
        if topk <= 0 or topk > scores.size:
            topk = scores.size

        scores, selections = jax.lax.top_k(scores, topk)
        locations = locations[selections]
        clss = clss[selections]

        threshold = 1 / distance_threshold / distance_threshold
        scores, locations, indices = non_max_suppression(
            scores,
            locations,
            output_size,
            threshold,
            score_threshold,
            return_selection=True,
        )

        idx_of_selected = jnp.argwhere(
            indices, size=output_size, fill_value=-1
        ).squeeze(-1)
        clss = jnp.where(idx_of_selected >= 0, clss[idx_of_selected], -1)

        locations = locations / jnp.asarray([self.z_scale, 1, 1]) # reverse z scaling

        return dict(
            scores=scores,
            locations=locations,
            classes=clss,
        )

    @nn.compact
    def _block(self, feature: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        conv_spec = self.conv_spec

        x = feature

        for n_ch in conv_spec:
            x = nn.Conv(n_ch, (3, 3))(x)
            x = self.normalization()(x)
            x = self.activation(x)

        self.sow("intermediates", "lpn_features", x)

        cls_logits = nn.Conv(self.detection_n_cls + 1, (1, 1))(x)
        regressions = nn.Conv(3, (1, 1))(x)

        # no z-localization for 2d input
        if x.shape[0] == 1:
            regressions = regressions.at[..., 0].set(0.)

        return dict(
            regressions=regressions,
            cls_logits=cls_logits,
        )

    def __call__(
        self,
        inputs: Sequence[ArrayLike],
        gt_locations: ArrayLike | None = None,
        gt_cls: ArrayLike | None = None,
        *,
        training: bool = False,
        **kwargs,
    ) -> dict:
        """
        Args:
            inputs: a list of features: Sequence[Array...]
            gt_locations: [N, 3]
            gt_cls: class ids. [N]

        Returns:
            A dict of model output
        """
        network_outputs = {}

        for lvl, scale in zip(self.feature_levels, self.feature_level_scales):
            feature = inputs[lvl]
            network_outputs[str(lvl)] = self._block(feature)
            network_outputs[str(lvl)]["scale"] = scale

        # prediction
        predictions = self._generate_predictions(network_outputs)

        # losses
        if training:
            losses = self._compute_losses(network_outputs, gt_locations, gt_cls)
            network_outputs["training_locations"] = self._best_match(
                gt_locations, 
                predictions["locations"],
            )

        else:
            losses = {}

        # format outputs
        return dict(
            detector=network_outputs,
            predictions=predictions,
            losses=losses,
        )
