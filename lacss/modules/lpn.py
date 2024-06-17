from __future__ import annotations

from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from ..losses.common import generalized_categorical_loss, binary_focal_crossentropy
from ..ops import sorted_non_max_suppression_indices
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
        max_output: final number of detection outputs 
        min_score: the minimal score required to be included in the output

        detection_roi: Parameter for label smoothing for focal loss
        detection_loss_gamma: gamma value for detection focal loss
        detection_loss_delta: delta value for offset regression huber loss
    """

    # network hyperparams
    feature_levels: Sequence[int] = (0, 1, 2)
    feature_level_scales: Sequence[int] = (4, 8, 16)
    conv_spec: Tuple[Sequence[int], Sequence[int]] = ((192, 192, 192, 192), ())

    detection_n_cls: int = 1

    # loss hyperparams
    detection_roi: float = 8.0
    detection_loss_gamma: float = 2.0
    detection_loss_delta: float = 1.0

    # detection hyperparams
    nms_threshold: float = 8.0
    pre_nms_topk: int = -1
    max_output: int = 1280
    min_score: float = 0.2

    def _to_targets(
        self,
        locations: ArrayLike,
        clses: ArrayLike,
        height: int,
        width: int,
        scale: float,
    ) -> tuple[Array, Array]:
        """Generate labels as LPN regression targets

        Returns:
            is_target: [H, W, n_cls] int32
            regression_target: [H, W, 2] float tensor
        """
        threshold_sq = (self.detection_roi / scale) ** 2

        flat_locations = locations / scale

        mesh = jnp.mgrid[:height, :width] + 0.5  # [2, H, W]
        distances = flat_locations[:, :, None, None] - mesh  # [N, 2, H, W]
        distances_sq = (distances * distances).sum(axis=1)  # [N, H, W]

        # masking off invalid
        distances_sq = jnp.where(
            flat_locations[:, None, None, 0] >= 0, distances_sq, float("inf")
        )

        indices = jnp.argmin(
            distances_sq, axis=0, keepdims=True
        )  # [1, H, W] map to nearest gt-loc
        cls_map = clses[
            indices.squeeze(0)
        ]  # [H, W] map to the cls of the nearest gt-loc
        # cls_map = jax.nn.one_hot(cls_map, self.detection_n_cls)  # [H, W, n_cls]

        best_distances = jnp.take_along_axis(distances_sq, indices, 0).squeeze(0)
        is_target = (
            best_distances < threshold_sq
        )  # [H, W] indicate whether the pixel is close to a gt-loc
        cls_target = jnp.where(is_target, cls_map, self.detection_n_cls)  # [H, W]

        indices = indices.repeat(2, axis=0)[None, ...]  # [1, 2, H, W]
        regression_target = jnp.take_along_axis(distances, indices, 0).squeeze(
            0
        )  # [2, H, W]
        regression_target = regression_target.transpose(1, 2, 0)  # [H, W, 2]

        return cls_target, regression_target

    def _compute_losses(self, network_outputs, gt_locations, gt_cls):
        cls_loss = []
        regr_loss = []

        for block_out in network_outputs.values():
            height, width, _ = block_out["cls_logits"].shape

            if gt_cls is None:
                gt_cls = jnp.zeros(gt_locations.shape[0], dtype=int)

            cls_target, regression_target = self._to_targets(
                gt_locations,
                gt_cls,
                height,
                width,
                block_out["scale"],
            )

            self.sow("intermediates", "cls_target", cls_target)
            self.sow("intermediates", "regression_target", regression_target)

            cls_loss.append(
                optax.softmax_cross_entropy_with_integer_labels(
                    block_out["cls_logits"],
                    cls_target,
                ).reshape(-1)
            )

            # cls_loss.append(
            #     binary_focal_crossentropy(
            #         jax.nn.sigmoid(block_out["cls_logits"]),
            #         jax.nn.one_hot(cls_target, self.detection_n_cls),
            #         gamma=self.detection_loss_gamma,
            #     ).mean(axis=-1)
            # )
            # p = jnp.take_along_axis(
            #     jax.nn.softmax(block_out["cls_logits"]),
            #     cls_target[..., None],
            #     axis=-1,
            # ).squeeze(-1)
            # cls_loss.append(
            #     binary_focal_crossentropy(p, jnp.ones_like(p),
            #         gamma=self.detection_loss_gamma,
            #     )
            # )

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

        cls_loss = jnp.concatenate(cls_loss).mean()
        regr_loss = jnp.concatenate(regr_loss).mean()

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

        # preprocess
        scores, locations, clss = [], [], []
        for block_out in network_outputs.values():
            logits = jax.nn.softmax(block_out["cls_logits"])
            scores_ = logits[..., :-1].sum(axis=-1)

            height, width = scores_.shape
            loc_ = jnp.mgrid[:height, :width] + 0.5
            loc_ = loc_.transpose(1, 2, 0) + block_out["regressions"]

            # set invalid location score to -1
            is_valid = (
                (loc_ > 0.0).all(axis=-1)
                & (loc_[:, :, 0] < height)
                & (loc_[:, :, 1] < width)
            )
            scores_ = jnp.where(is_valid, scores_, -1)

            loc_ = loc_ * block_out["scale"]  # scale to original resolution
            clss_ = logits[..., :-1].argmax(axis=-1)

            scores.append(scores_.reshape(-1))
            locations.append(loc_.reshape(-1, 2))
            clss.append(clss_.reshape(-1))

        scores, locations, clss = map(jnp.concatenate, (scores, locations, clss))

        # sort and nms
        if topk <= 0 or topk > scores.size:
            topk = scores.size

        scores, selections = jax.lax.top_k(scores, topk)
        locations = locations[selections]
        clss = clss[selections]

        threshold = 1 / distance_threshold / distance_threshold
        indices = sorted_non_max_suppression_indices(
            scores,
            locations,
            output_size,
            threshold,
            score_threshold,
        )
        idx_of_selected = jnp.argwhere(
            indices, size=output_size, fill_value=-1
        ).squeeze(-1)

        scores = jnp.where(idx_of_selected >= 0, scores[idx_of_selected], -1.0)
        locations = jnp.where(
            idx_of_selected[:, None] >= 0, locations[idx_of_selected], -1.0
        )
        clss = jnp.where(idx_of_selected >= 0, clss[idx_of_selected], -1)

        return dict(
            scores=scores,
            locations=locations,
            classes=clss,
        )

    @nn.compact
    def _block(self, feature: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        conv_spec = self.conv_spec

        x = feature

        for n_ch in conv_spec[0]:
            x = nn.Conv(n_ch, (3, 3), use_bias=False)(x)
            x = nn.LayerNorm(use_scale=False)(x)
            # x = nn.GroupNorm(num_groups=n_ch)(x[None, ...])[0]
            x = jax.nn.relu(x)

        for n_ch in conv_spec[1]:
            x = nn.Conv(n_ch, (1, 1), use_bias=False)(x)
            x = nn.LayerNorm(use_scale=False)(x)
            # x = nn.GroupNorm(num_groups=n_ch)(x[None, ...])[0]
            x = jax.nn.relu(x)

        cls_logits = nn.Conv(self.detection_n_cls + 1, (1, 1))(x)
        regressions = nn.Conv(2, (1, 1))(x)

        self.sow("intermediates", "lpn_features", x)

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
    ) -> dict:
        """
        Args:
            inputs: a list of features: Sequence[Array...]
            gt_locations: [N, 2]
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
        else:
            losses = {}

        # format outputs
        return dict(
            lpn=network_outputs,
            predictions=predictions,
            losses=losses,
        )
