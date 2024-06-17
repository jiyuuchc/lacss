from __future__ import annotations

import pickle

import ml_collections
import optax

import livecell_dataset
from lacss.modules import FPN, LPN, ConvNeXt, Lacss, Segmentor

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "cnsp4_bf"

    config.dataset = livecell_dataset.get_config()

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.batchsize = 3
    config.train.n_steps = 100000
    config.train.validation_interval = 10000

    lr = optax.piecewise_constant_schedule(0.0005, {90000: 0.1})
    config.train.optimizer = optax.adamw(lr)

    # config.model = Lacss.get_preconfigued("default")
    patch_size = 4
    model = Lacss(
        backbone=ConvNeXt.get_model_small(patch_size=patch_size),
        integrator=FPN(256),
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
    config.model = model

    return config
