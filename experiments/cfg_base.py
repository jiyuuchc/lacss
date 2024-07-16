from __future__ import annotations

import ml_collections
import combined_dataset

from lacss.modules import Lacss

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "lacss_base"

    config.fpn_dim = 384

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.batchsize = 3
    config.train.steps = 100000
    config.train.validation_interval = 10000
    config.train.lr = 5e-5
    config.train.weight_decay = 1e-3

    config.model = Lacss.get_small_model().get_config()

    data = combined_dataset.get_config()
    config.data = ml_collections.ConfigDict()
    config.data.ds_train = data.ds_train.repeat().batch(config.train.batchsize).prefetch(1)
    config.data.ds_val = dict(
        livecell = data.ds_livecell_val.prefetch(1),
        cellpose = data.ds_cellpose_val.prefetch(1),
    )

    return config
