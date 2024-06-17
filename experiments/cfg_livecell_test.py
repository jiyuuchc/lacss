from __future__ import annotations

import ml_collections
import optax
import tensorflow as tf

import livecell_dataset
from lacss.modules import Lacss

tf.config.set_visible_devices([], "GPU")

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "livecell_test"

    config.dataset = livecell_dataset.get_config()

    config.train = ml_collections.ConfigDict()
    config.train.seed = 42
    config.train.batchsize = 3
    config.train.n_steps = 100000
    config.train.validation_interval = 10000

    lr = optax.piecewise_constant_schedule(0.0005, {90000: 0.1})
    config.train.optimizer = optax.adamw(lr, weight_decay=1e-3)

    config.model = Lacss.get_preconfigued("small")

    return config
