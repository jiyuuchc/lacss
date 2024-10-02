from __future__ import annotations

from pathlib import Path
from functools import partial

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds

from lacss.data.utils import gf_batch
from lacss.modules.common import picklable_relu
from cfg_2d import augment, check_cell_number, format_train_data, format_test_data

DATAPATH = Path("/home/FCAM/jyu/datasets")
IMG_SIZE = [544, 544]

DATASET="livecell"
#DATASET="cellpose:1.0.0"

def get_config(model="tiny"):
    import lacss.data
    from lacss.modules import Lacss
    import flax.linen as nn

    config = ml_collections.ConfigDict()
    config.name = "lacss_livecell_test"

    config.train = ml_collections.ConfigDict()
    config.train.seed = 42
    config.train.steps = 100000
    config.train.finetune_steps = 20000
    config.train.validation_interval = 10000
    config.train.lr = 1e-4
    config.train.weight_decay = 0.05

    config.train.config = ml_collections.ConfigDict()
    config.train.config.detection_roi = 8
    config.train.config.similarity_score_scaling = 4
    config.train.config.n_labels_min = 1
    config.train.config.n_labels_max = 25

    config.train.backbone_dropout = 0.4

    config.model = Lacss.get_preconfigued(model)
    config.model.detector.max_output = 1024
    config.model.backbone.activation = picklable_relu
    # config.model.backbone.drop_path_rate = 0.6

    livecell_train = (
        tfds.load(DATASET, split="train")
        .repeat()
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        # .ragged_batch(4)
        # .map(lacss.data.mosaic, num_parallel_calls=tf.data.AUTOTUNE)
        # .map(partial(lacss.data.random_crop_or_pad, target_size=IMG_SIZE, area_ratio_threshold=0.5),
        #     num_parallel_calls=tf.data.AUTOTUNE)
        .map(
            partial(lacss.data.cutout, size=10, n=50),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .filter(check_cell_number)
        .map(format_train_data)
    )

    config.data = ml_collections.ConfigDict()
    config.data.ds_train = livecell_train
    config.data.ds_val = tfds.load(DATASET, split="val").map(format_test_data)
    config.data.batch_size = 3

    return config
