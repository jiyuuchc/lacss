#!/usr/bin/env python

from logging.config import valid_ident
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

import json, os
from functools import partial
from os.path import join

import optax, jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

import lacss
try:
    from . import data
except:
    import data

import typer
app = typer.Typer(pretty_exceptions_enable=False)

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def train_parser_semisupervised(inputs):
    cell_type = inputs['cell_type']
    inputs = lacss.data.parse_train_data_func(inputs, size_jitter=(0.85, 1.15), target_height=544, target_width=544)

    image = inputs['image']
    gt_locations = inputs['locations']
    image_mask = tf.cast(inputs['binary_mask'], tf.float32)
    group_num = tf.cast(cell_type, tf.float32)

    image = tf.image.random_contrast(image, 0.6, 1.4)
    image = tf.image.random_brightness(image, 0.3)

    x_data = (image, gt_locations, group_num)
    y_data = dict(binary_mask = image_mask)
    return x_data, y_data

def val_parser(inputs):
    return (dict(
            image = inputs['image'],
        ), dict(
            gt_boxes = inputs['bboxes'],
            gt_locations = inputs['locations'],
        ))

def prepare_data(datapath, celltype, batchsize, n_buckets = 16):
    ds_train = data.livecell_dataset_from_tfrecord(join(datapath, 'train.tfrecord'))

    if celltype >= 0:
        ds_train = ds_train.filter(lambda x: x['cell_type'] == celltype)

    ds_train = ds_train.map(train_parser_semisupervised)
    ds_train = ds_train.filter(lambda x, _: tf.size(x[1])>0).repeat()
    ds_train = ds_train.bucket_by_sequence_length(
        element_length_func = lambda x, y: tf.shape(x[1])[0],
        bucket_boundaries = list(np.arange(1, n_buckets) * (4096 // n_buckets) + 1),
        bucket_batch_sizes = (batchsize,) * n_buckets,
        padding_values = -1.0,
        pad_to_bucket_boundary = True,
    )

    ds_val = data.livecell_dataset_from_tfrecord(join(datapath, 'val.tfrecord'))
    ds_val = ds_val.map(val_parser).batch(1)

    return ds_train, ds_val

def get_model(cmd, config, batchsize, seed):
    n_epochs = 30
    steps_per_epoch = 3500
    optimizer = optax.adam(0.0005)

    if cmd == 'resume':
        cp = lacss.trainer.Trainer.from_checkpoint(config)
        assert(isinstance(cp, lacss.trainer.Trainer))
        trainer = cp

        print(f'Loaded checkpoint {config}')
        try:
            init_epoch = int(config.split('-')[-1])
        except:
            init_epoch = 0
    else:
        if cmd == 'transfer':
            import pickle
            model = pickle.load(open(config, 'rb'))
            assert isinstance(model, lacss.modules.LacssWithHelper)
        elif cmd == 'config':
            with open(config) as f:
                model_cfg = json.load(f)
            model = lacss.modules.LacssWithHelper(**model_cfg)
        else:
            raise ValueError('Cmd musst be one of "resume", "transfer" or "config"')

        loss = [
            lacss.losses.LPNLoss(),
            lacss.losses.AuxEdgeLoss(),
            lacss.losses.InstanceLoss(),
        ]

        trainer = lacss.train.Trainer(
            model = model,
            optimizer = optimizer,
            losses = loss,
            seed = seed,
        )
        init_epoch = 0

    return trainer, init_epoch, n_epochs, steps_per_epoch

@app.command()
def run_training(
    cmd: str,
    config: str,
    datapath: str = '../livecell_dataset',
    logpath: str = './',
    celltype: int = -1,
    seed: int = 42,
    batchsize: int = 1,
):
    tf.random.set_seed(seed)

    try:
        os.makedirs(logpath)
    except:
        pass

    ds_train, ds_val = prepare_data(datapath, celltype, batchsize)
    print(ds_train.element_spec)

    trainer, init_epoch, n_epochs, steps_per_epoch = get_model(cmd, config, batchsize, seed)
    print(json.dumps(trainer.model.get_config(), indent=2, sort_keys=True))

    epoch = init_epoch
    train_gen = lacss.trainer.TFDatasetAdapter(ds_train, steps=-1).get_dataset()
    val_gen = lacss.trainer.TFDatasetAdapter(ds_val).get_dataset()
    val_metrics = [
        lacss.metrics.LoiAP([0.1, 0.2, 0.5, 1.0]),
        lacss.metrics.BoxAP([0.5, 0.75])
    ]
    for steps, logs in enumerate(trainer(train_gen)):
        if (steps + 1) % steps_per_epoch == 0:
            epoch += 1
            print(f'epoch - {epoch}')
            print(", ".join([f'{k}:{v:.4f}' for k,v in logs.items()]))

            trainer.checkpoint(join(logpath, f'cp-{epoch}'))
            trainer.reset()

            var_logs = trainer.test_and_compute(val_gen, strategy=lacss.train.Core)

            print(", ".join([f'{k}:{v}' for k,v in var_logs.items()]))

            if epoch == n_epochs:
                break

if __name__ =="__main__":
    app()
