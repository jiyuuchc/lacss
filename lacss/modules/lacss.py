from functools import partial
import typing as tp
from dataclasses import asdict, field

import jax
import jax.numpy as jnp
import flax.linen as nn

from . import ResNet, ConvNeXt, Segmentor, LPN, LPN, Detector
from .auxiliary import *
from ..ops import *

class Lacss(nn.Module):
    backbone: type = "ResNet"
    backbone_cfg: dict = field(default_factory=dict)
    lpn:dict = field(default_factory=dict)
    detector: dict = field(default_factory=dict)
    segmentor: dict = field(default_factory=dict)

    def setup(self):
        cls = globals()[self.backbone]
        self._backbone = cls(**self.backbone_cfg)
        self._lpn = LPN(**self.lpn)
        self._detector = Detector(**self.detector)
        self._segmentor = Segmentor(**self.segmentor)

    def get_config(self):
        return dict(
            backbone = self.backbone,
            backbone_config = asdict(self._backbone),
            lpn = asdict(self._lpn),
            detector = asdict(self._detector),
            segmentor = asdict(self._segmentor),
        )

    @classmethod
    def from_config(cls, config):
        if 'backbone' in config:
            backbone_cls = dict(ResNet = ResNet, ConvNeXt = ConvNeXt)
            config['backbone'] = backbone_cls[config.pop('backbone')]

        obj = cls(**config)
        return obj

    def __call__(
        self, 
        image: jnp.ndarray, 
        gt_locations: jnp.ndarray = None,
        *,
        training: bool = None,
    ) -> dict:
        '''
        Args:
            image: [N, H, W, C]
            gt_locations: [N, M, 2] if training, otherwise None
        Returns:
            a dict of model outputs
        '''
        n_batches, orig_height, orig_width, ch = image.shape
        if ch == 1:
            image = jnp.repeat(image, 3, axis=-1)
        elif ch == 2:
            image = jnp.concatenate([image, jnp.zeros_like(image[...,:1])], axis=-1)

        assert image.shape[-1] == 3

        # ensure input size is multiple of 32
        height = ((orig_height-1) // 32 + 1) * 32
        width = ((orig_width-1) // 32 + 1) * 32
        image = jnp.pad(image, [[0,0],[0, height-orig_height],[0, width - orig_width],[0,0]])
 
        # backbone
        encoder_features, features = self._backbone(image, training=training)
        model_output = dict(
            encoder_features = encoder_features,
            decoder_features = features,
        )

        # detection
        scaled_gt_locations = gt_locations / jnp.array([height, width]) if gt_locations is not None else None
        model_output.update(self._lpn(
            inputs = features, 
            scaled_gt_locations = scaled_gt_locations,
            training=training,
        ))
        model_output.update(self._detector(
            scores = model_output['lpn_scores'], 
            regressions = model_output['lpn_regressions'],
            gt_locations = gt_locations,
            training = training,
            ))

        # segmentation
        locations = model_output['training_locations' if training else 'pred_locations']
        scaled_locs = locations / jnp.array([height, width])
        model_output.update(self._segmentor(
            features = features, 
            locations = scaled_locs,
        ))

        return model_output

class LacssWithHelper(nn.Module):
    cfg: dict = field(default_factory=dict)
    aux_edge_cfg: tp.Optional[dict] = None
    aux_fg_cfg: tp.Optional[dict] = None

    @nn.compact
    def __call__(self, image, gt_locations=None, category=None, *, training=None):

        preds = Lacss(**self.cfg)(image, gt_locations, training=training)

        if self.aux_edge_cfg is not None:
            preds.update(dict(
                edge_pred = AuxInstanceEdge(**self.aux_edge_cfg)(image, category)
            ))
        if self.aux_fg_cfg is not None:
            preds.update(dict(
                fg_pred = AuxForeground(**self.aux_fg_cfg)(image, category)
            ))

        return preds
