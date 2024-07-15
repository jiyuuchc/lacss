from __future__ import annotations

from dataclasses import field

import flax.linen as nn
import jax.numpy as jnp

from ..ops import *
from ..typing import *
from ..utils import deep_update
from .convnext import ConvNeXt
from .lpn import LPN
from .segmentor import Segmentor
from .stack_integrator import StackIntegrator

class Lacss(nn.Module):
    """Main class for LACSS model

    Attributes:
        stem: Prepprocessing module
        backbone: by default a ConvNeXt CNN
        integrator: module for integrating backbone output
        detector: detection head to predict cell locations
        segmentor: The segmentation head
    """

    stem: nn.Module | None = None
    backbone: nn.Module = field(default_factory=ConvNeXt)
    integrator: nn.Module | None = field(default_factory=StackIntegrator)
    detector: nn.Module = field(default_factory=LPN)
    segmentor: nn.Module | None = field(default_factory=Segmentor)

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
            gt_locations: [M, 2/3] if training, otherwise None
            gt_cls: [M] instance classes for multi-class prediction
        Returns:
            a dict of model outputs

        """
        outputs = {}

        x = image

        # if inputs are 2D
        if x.ndim == 3:
            x = x[None, ...]

        assert x.ndim == 4

        if gt_locations is not None:
            if gt_locations.shape[-1] == 2:
                gt_locations = jnp.c_[jnp.zeros_like(gt_locations[:, :1]) + .5, gt_locations]
        
            if gt_cls is None:
                gt_cls = jnp.zeros(gt_locations.shape[0], dtype=int)

            assert x.ndim == 4
            assert gt_locations.ndim == 2 and gt_locations.shape[-1] == 3

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
            if training:
                seg_locs= outputs["detector"]["training_locations"]
                seg_cls = gt_cls
            elif gt_locations is not None: # not training, but locations of cells are known
                seg_locs, seg_cls = gt_locations, gt_cls
            else:
                seg_locs = outputs["predictions"]["locations"]
                seg_cls = outputs["predictions"]["classes"]

            self.sow("intermediates", "segmentation_locations", seg_locs)
            self.sow("intermediates", "segmentation_cls", seg_cls)

            segmentor_out = self.segmentor(x, seg_locs, seg_cls, training=training)

            outputs = deep_update(outputs, segmentor_out)

        return outputs

    def get_config(self):
        from dataclasses import asdict
        from ..utils import remove_dictkey

        cfg = asdict(self)

        remove_dictkey(cfg, "parent")

        return cfg

    @classmethod
    def from_config(cls, config):
        from collections import defaultdict             
        config_ = defaultdict(lambda: {})
        config_.update(config)
        return Lacss(
            backbone=ConvNeXt(**config_["backbone"]),
            integrator=StackIntegrator(**config_["integrator"]),
            detector=LPN(**config_["detector"]),
            segmentor=Segmentor(**config_["segmentor"]),
        )

    @classmethod
    def get_default_model(cls, patch_size=4):
        if not patch_size in (1, 2, 4):
            raise ValueError("patch_size must be 1, 2 or 4")
        return cls(
            backbone=ConvNeXt.get_model_small(patch_size=patch_size),
            integrator=StackIntegrator(dim_out=384),
            detector=LPN(
                conv_spec=(256, 256, 256, 256),
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
            integrator=StackIntegrator(dim_out=256),
            detector=LPN(
                conv_spec=(192, 192, 192, 192),
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
            integrator=StackIntegrator(dim_out=512),
            detector=LPN(
                conv_spec=(384, 384, 384, 384),
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
