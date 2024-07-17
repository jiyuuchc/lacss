from __future__ import annotations

import ml_collections

def get_config():
    import tensorflow as tf
    from combined_dataset import (
        ds_cellpose, ds_a431, ds_livecell, ds_tn,
        ds_livecell_val, ds_cellpose_val,
    )
    from dataset3d import ds_mayu, ds_c3dl, ds_n3dh
    from lacss.modules import Lacss

    config = ml_collections.ConfigDict()
    config.name = "lacss_base"

    config.fpn_dim = 512

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.batchsize = 3
    config.train.steps = 150000
    config.train.finetune_steps = 50000
    config.train.validation_interval = 10000
    config.train.lr = 1e-4
    config.train.weight_decay = 1e-3

    config.model = Lacss.get_small_model().get_config()
    config.model.segmentor.encoder_dims = (16, 8, 8, 4)

    ds_s = tf.data.Dataset.sample_from_datasets(
        [ds_cellpose.repeat(), ds_a431.repeat(), ds_livecell.repeat()],
        [.25, .1, .65],
    ).batch(config.train.batchsize).prefetch(1)

    ds_ws = ds_tn.map(lambda x,y: x).repeat().batch(config.train.batchsize).prefetch(1)

    ds_mayu = ds_mayu.repeat().batch(1).prefetch(1)
    ds_c3dl = ds_c3dl.repeat().batch(1).prefetch(1)
    ds_n3dh = ds_n3dh.repeat().batch(1).prefetch(1)

    ds_train = dict(
        datasets = (ds_s, ds_ws, ds_mayu, ds_c3dl, ds_n3dh),
        weights = (1, 0.1, .5, 0.2, 0.1),
    )

    ds_val = dict(
        livecell = ds_livecell_val.prefetch(1),
        cellpose = ds_cellpose_val.prefetch(1),
    )

    config.data = ml_collections.ConfigDict()
    config.data.ds_train = ds_train
    config.data.ds_val = ds_val

    return config
