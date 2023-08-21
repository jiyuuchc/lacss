from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, field

import flax.linen as nn
import jax.numpy as jnp

from ..ops import *
from ..typing import *
from .convnext import ConvNeXt
from .detector import Detector
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

    backbone: ConvNeXt = field(default_factory=ConvNeXt)
    lpn: LPN = field(default_factory=LPN)
    detector: Detector = field(default_factory=Detector)
    segmentor: Segmentor = field(default_factory=Segmentor)

    @classmethod
    def from_config(cls, config: dict):
        """Factory method to build an Lacss instance from a configuration dictionary

        Args:
            config: A configuration dictionary.

        Returns:
            An Lacss instance.

        """
        config_ = defaultdict(lambda: {})
        config_.update(config)
        return Lacss(
            backbone=ConvNeXt(**config_["backbone"]),
            lpn=LPN(**config_["lpn"]),
            detector=Detector(**config_["detector"]),
            segmentor=Segmentor(**config_["segmentor"]),
        )
        # return dataclass_from_dict(cls, config)

    def get_config(self) -> dict:
        """Convert to a configuration dict. Can be serialized with json

        Returns:
            config: a configuration dict
        """
        return asdict(self)

    def __call__(
        self,
        image: ArrayLike,
        gt_locations: ArrayLike | None = None,
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
        orig_height, orig_width, ch = image.shape
        if ch == 1:
            image = jnp.repeat(image, 3, axis=-1)
        elif ch == 2:
            image = jnp.concatenate([image, jnp.zeros_like(image[..., :1])], axis=-1)

        # assert image.shape[-1] == 3

        # ensure input size is multiple of 32
        height = ((orig_height - 1) // 32 + 1) * 32
        width = ((orig_width - 1) // 32 + 1) * 32
        image = jnp.pad(
            image, [[0, height - orig_height], [0, width - orig_width], [0, 0]]
        )

        # backbone
        encoder_features, features = self.backbone(image, training=training)
        model_output = dict(
            encoder_features=encoder_features,
            decoder_features=features,
        )

        # detection
        scaled_gt_locations = (
            gt_locations / jnp.array([height, width])
            if gt_locations is not None
            else None
        )
        model_output.update(
            self.lpn(
                inputs=features,
                scaled_gt_locations=scaled_gt_locations,
            )
        )

        if gt_locations is not None or not training:

            model_output.update(
                self.detector(
                    scores=model_output["lpn_scores"],
                    regressions=model_output["lpn_regressions"],
                    gt_locations=gt_locations,
                    training=training,
                )
            )

            # segmentation
            locations = model_output[
                "training_locations" if training else "pred_locations"
            ]
            scaled_locs = locations / jnp.array([height, width])
            model_output.update(
                self.segmentor(
                    features=features,
                    locations=scaled_locs,
                )
            )

        return model_output
