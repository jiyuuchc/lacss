from __future__ import annotations

from dataclasses import field

import flax.linen as nn
import jax.numpy as jnp

from ..ops import *
from ..typing import *
from ..utils import deep_update
from .common import FPN
from .convnext import ConvNeXt
from .lpn import LPN
from .segmentor import Segmentor


class Lacss(nn.Module):
    """Main class for LACSS model

    Attributes:
        backbone: The ConvNeXt backbone
        lpn: The LPN head for detecting cell location
        detector: A weight-less module interpreting lpn output
        segmentor: The segmentation head

    """

    stem: nn.Module | None = None
    backbone: nn.Module = field(default_factory=ConvNeXt)
    integrator: nn.Module | None = field(default_factory=FPN)
    detector: nn.Module = field(default_factory=LPN)
    segmentor: nn.Module | None = field(default_factory=Segmentor)

    max_proposal_offset: float = 12.0

    def _best_match(self, gt_locations, pred_locations, height, width):
        """replacing gt_locations with pred_locations if the close enough
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
        image: ArrayLike,
        gt_locations: ArrayLike | None = None,
        gt_cls: ArrayLike | None = None,
        *,
        training: bool | None = None,
    ) -> dict:
        """
        Args:
            image: [H, W, C]
            gt_locations: [M, 2] if training, otherwise None
        Returns:
            a dict of model outputs

        """
        height, width = image.shape[:2]

        outputs = {}

        x = image
        if self.stem is not None:
            x = self.stem(x, training=training)

        x = self.backbone(x, training=training)
        self.sow("intermediates", "encoder_features", x)

        if self.integrator is not None:
            x = self.integrator(x, training=training)
            self.sow("intermediates", "decoder_features", x)

        detector_out = self.detector(x, gt_locations, gt_cls, training=training)
        outputs = deep_update(outputs, detector_out)

        if self.segmentor is not None:
            pred_locs = outputs["predictions"]["locations"]
            pred_cls = outputs["predictions"]["classes"]

            if gt_cls is None and gt_locations is not None:
                gt_cls = jnp.zeros(gt_locations.shape[0], dtype=int)

            if training:
                seg_locs, seg_cls = (
                    self._best_match(gt_locations, pred_locs, height, width),
                    gt_cls,
                )
            elif gt_locations is not None:
                seg_locs, seg_cls = gt_locations, gt_cls
            else:
                seg_locs, seg_cls = pred_locs, pred_cls

            self.sow("intermediates", "segmentation_locations", seg_locs)
            self.sow("intermediates", "segmentation_cls", seg_cls)

            segmentor_out = self.segmentor(x, seg_locs, seg_cls)

            outputs = deep_update(outputs, segmentor_out)

        return outputs

    @classmethod
    def get_default_model(cls, patch_size=4):
        if not patch_size in (1, 2, 4):
            raise ValueError("patch_size must be 1, 2 or 4")
        return cls(
            backbone=ConvNeXt.get_model_small(patch_size=patch_size),
            integrator=FPN(384),
            detector=LPN(
                conv_spec=((256, 256, 256, 256), ()),
                feature_levels=(0, 1, 2),
                feature_level_scales=(patch_size, patch_size * 2, patch_size * 4),
            ),
            segmentor=Segmentor(
                conv_spec=((256, 256, 256), (64,)),
                feature_level=0,
                feature_scale=patch_size,
            ),
        )

    @classmethod
    def get_small_model(cls, patch_size=4):
        return cls(
            backbone=ConvNeXt.get_model_tiny(patch_size=patch_size),
            integrator=FPN(256),
            detector=LPN(
                conv_spec=((192, 192, 192, 192), ()),
                feature_levels=(0, 1, 2),
                feature_level_scales=(patch_size, patch_size * 2, patch_size * 4),
            ),
            segmentor=Segmentor(
                conv_spec=((192, 192, 192), (48,)),
                feature_level=0,
                feature_scale=patch_size,
            ),
        )

    @classmethod
    def get_large_model(cls, patch_size=4):
        return cls(
            backbone=ConvNeXt.get_model_base(patch_size=patch_size),
            integrator=FPN(512),
            detector=LPN(
                conv_spec=((384, 384, 384, 384), ()),
                feature_levels=(0, 1, 2),
                feature_level_scales=(patch_size, patch_size * 2, patch_size * 4),
            ),
            segmentor=Segmentor(
                conv_spec=((384, 384, 384), (128,)),
                feature_level=0,
                feature_scale=patch_size,
            ),
        )

    @classmethod
    def get_preconfigued(cls, config: str, *, patch_size: int = 4):
        if config == "default" or config == "base":
            return cls.get_default_model(patch_size)
        elif config == "small":
            return cls.get_small_model(patch_size)
        elif config == "large":
            return cls.get_large_model(patch_size)
        else:
            raise ValueError(f"Unkown model config {config}")
