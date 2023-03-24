import typing as tp
from dataclasses import asdict, field
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..ops import *
from .auxiliary import *
from .convnext import ConvNeXt
from .detector import Detector
from .lpn import LPN
from .resnet import ResNet
from .segmentor import Segmentor


class Lacss(nn.Module):
    backbone: type = "ResNet"
    backbone_cfg: dict = field(default_factory=dict)
    lpn: dict = field(default_factory=dict)
    detector: dict = field(default_factory=dict)
    segmentor: dict = field(default_factory=dict)

    def setup(self):
        cls = globals()[self.backbone]
        self._backbone = cls(**self.backbone_cfg)
        self._lpn = LPN(**self.lpn)
        self._detector = Detector(**self.detector)
        self._segmentor = Segmentor(**self.segmentor)

    def __call__(
        self,
        image: jnp.ndarray,
        gt_locations: jnp.ndarray = None,
        *,
        training: bool = None,
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

        assert image.shape[-1] == 3

        # ensure input size is multiple of 32
        height = ((orig_height - 1) // 32 + 1) * 32
        width = ((orig_width - 1) // 32 + 1) * 32
        image = jnp.pad(
            image, [[0, height - orig_height], [0, width - orig_width], [0, 0]]
        )

        # backbone
        encoder_features, features = self._backbone(image, training=training)
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
            self._lpn(
                inputs=features,
                scaled_gt_locations=scaled_gt_locations,
            )
        )
        model_output.update(
            self._detector(
                scores=model_output["lpn_scores"],
                regressions=model_output["lpn_regressions"],
                gt_locations=gt_locations,
                training=training,
            )
        )

        # segmentation
        locations = model_output["training_locations" if training else "pred_locations"]
        scaled_locs = locations / jnp.array([height, width])
        model_output.update(
            self._segmentor(
                features=features,
                locations=scaled_locs,
            )
        )

        return model_output


class LacssWithHelper(nn.Module):
    cfg: dict = field(default_factory=dict)
    aux_edge_cfg: tp.Optional[dict] = None
    aux_fg_cfg: tp.Optional[dict] = None

    def setup(self):
        self._lacss = Lacss(**self.cfg)
        if self.aux_edge_cfg is not None:
            self._aux_edge_module = AuxInstanceEdge(**self.aux_edge_cfg)
        if self.aux_fg_cfg is not None:
            self._aux_fg_module = AuxForeground(**self.aux_fg_cfg)

    def __call__(self, image, gt_locations=None, category=None, *, training=False):

        preds = self._lacss(image, gt_locations, training=training)

        if self.aux_edge_cfg is not None:

            preds.update(self._aux_edge_module(image, category=category))

        if self.aux_fg_cfg is not None:

            preds.update(
                self._aux_fg_module(image, category=category, augment=training)
            )

        return preds
