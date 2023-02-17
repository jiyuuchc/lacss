#!/usr/bin/env python

from logging.config import valid_ident
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
from functools import partial
import os
from os.path import join

import numpy as np
import optax
import elegy as eg
import jax
jnp = jax.numpy

import lacss
try:
    from . import data
except:
    import data

import typer
app = typer.Typer(pretty_exceptions_enable=False)

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def cb_fn(epoch, logs, model, ds):

    def pad_to_block_size(x, block_size=256, constant_values=-1.):
        ns = ((x.shape[0] - 1) // block_size + 1) * block_size
        padding = ns - x.shape[0]
        return jnp.pad(x, [[0,padding],[0,0]], constant_values=constant_values)

    metrics = eg.metrics.Metrics([
        lacss.metrics.LoiAP([0.1, 0.2, 0.5, 1.0]),
        lacss.metrics.BoxAP([0.5, 0.75])
    ])
    
    for data in ds:
        inputs, labels = jax.tree_map(lambda v: jnp.asarray(v), data)
        labels = jax.tree_map(jax.vmap(pad_to_block_size), labels)
        preds = model.predict_on_batch(**inputs)
        metrics.update(preds=preds, **labels)
        
    logs.update(metrics.compute())

def train_parser_supervised(inputs):
    inputs = lacss.data.parse_train_data_func_full_annotation(inputs, target_height=544, target_width=544)
    
    image = inputs['image']
    gt_locations = inputs['locations']
    mask_labels = tf.cast(inputs['mask_labels'], tf.float32)

    if tf.random.uniform([]) >= .5:
        image = tf.image.transpose(image)
        gt_locations = gt_locations[..., ::-1]
        mask_labels = tf.transpose(mask_labels)

    x_data = dict(
        image = image,
        gt_locations = gt_locations,
    )
    y_data = dict(
        mask_labels = mask_labels,
    )

    return x_data, y_data

def train_parser_semisupervised(inputs):
    cell_type = inputs['cell_type']
    inputs = lacss.data.parse_train_data_func(inputs, size_jitter=(0.85, 1.15), target_height=544, target_width=544)

    image = inputs['image']
    gt_locations = inputs['locations']
    image_mask = tf.cast(inputs['binary_mask'], tf.float32)
    group_num = tf.cast(cell_type, tf.float32)

    image = tf.image.random_contrast(image, 0.6, 1.4)
    image = tf.image.random_brightness(image, 0.3)

    x_data = dict(
        image = image,
        gt_locations = gt_locations,
    )
    y_data = dict(
        binary_mask = image_mask,
        group_num = group_num,
    )

    return x_data, y_data

def val_parser(inputs):
    return (dict(
            image = inputs['image'],
        ), dict(
            gt_boxes = inputs['bboxes'],
            gt_locations = inputs['locations'],
        ))

def prepare_data(datapath, celltype, supervised, batchsize, n_buckets = 16):
    ds_train = data.livecell_dataset_from_tfrecord(join(datapath, 'train.tfrecord'))

    if celltype >= 0:
        ds_train = ds_train.filter(lambda x: x['cell_type'] == celltype)

    train_parser = train_parser_supervised if supervised else train_parser_semisupervised
    ds_train = ds_train.map(train_parser)
    ds_train = ds_train.filter(lambda x, _: tf.size(x['gt_locations'])>0).repeat()    
    ds_train = ds_train.bucket_by_sequence_length(
        element_length_func = lambda x, y: tf.shape(x['gt_locations'])[0],
        bucket_boundaries = list(np.arange(1, n_buckets) * (4096 // n_buckets) + 1),
        bucket_batch_sizes = (batchsize,) * n_buckets,
        padding_values = -1.0,
        pad_to_bucket_boundary = True,
    )

    ds_val = data.livecell_dataset_from_tfrecord(join(datapath, 'val.tfrecord'))
    ds_val = ds_val.map(val_parser).batch(1)

    return ds_train, ds_val

def get_schedule(schedule, batchsize):
    if schedule == 1:
        n_epochs = 30
        steps_per_epoch = 3500
        optimizer = optax.adam(0.0005)
    elif schedule == 2:
        n_epochs = 10
        steps_per_epoch = 3500 * 5
        schedules = [optax.cosine_decay_schedule(0.002 * batchsize, steps_per_epoch)] * n_epochs
        boundaries = [steps_per_epoch * i for i in range(1, n_epochs)]
        schedule = optax.join_schedules(schedules, boundaries)
        optimizer = optax.adamw(schedule)
    else:
        raise ValueError()

    return n_epochs, steps_per_epoch, optimizer

def get_model(resume, transfer, config, supervised, seed, optimizer):
    if resume is not None:
        model = eg.model.model_base.load(resume)
        print(f'Loaded checkpoint {resume}')
        try:
            init_epoch = int(resume.split('-')[-1])
        except:
            init_epoch = 0
        return model, init_epoch        

    if transfer is not None:
        model = eg.model.model_base.load(transfer)
        module = model.module
        module.freeze(False, inplace=True)
    elif config is not None:
        with open(config) as f:
            model_cfg = json.load(f)
        module = lacss.modules.Lacss.from_config(model_cfg)
    else:
        raise ValueError('Must specify at least one of the "--resume", "--transfer" or "--config"')

    loss = [
        lacss.losses.DetectionLoss(),
        lacss.losses.LocalizationLoss(),
    ]
    if not supervised:
        loss.append(lacss.losses.InstanceEdgeLoss())
        loss.append(lacss.losses.InstanceLoss())
    else:
        loss.append(lacss.losses.SupervisedInstanceLoss())
        pass

    model = eg.Model(
        module = module,     
        optimizer = optimizer,
        seed = seed,
        loss = loss,
    )
    init_epoch = 0

    return model, init_epoch

@app.command()
def run_training(
    datapath: str,
    logpath: str = './',
    config: str = None,
    resume: str = None,
    transfer: str = None,
    celltype: int = -1,
    supervised: bool = False,
    seed: int = 42,
    verbose: int = 2,
    schedule: int = 1,
    batchsize: int = 1,
):
    tf.random.set_seed(seed)

    try:
        os.makedirs(logpath)
    except:
        pass

    ds_train, ds_val = prepare_data(datapath, celltype, supervised, batchsize)
    print(ds_train.element_spec)

    n_epochs, steps_per_epoch, optimizer = get_schedule(schedule, batchsize)
    model, init_epoch = get_model(resume, transfer, config, supervised, seed, optimizer)

    print(json.dumps(model.module.get_config(), indent=2, sort_keys=True))

    callbacks = [
        eg.callbacks.TensorBoard(logpath),
        eg.callbacks.ModelCheckpoint(path=join(logpath, 'chkpt-{epoch:02d}')),
        eg.callbacks.LambdaCallback(on_epoch_end = partial(cb_fn, model=model, ds=ds_val)),
    ]
    model.fit(
        inputs=ds_train, 
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=init_epoch,
        verbose = verbose,
        callbacks = callbacks,
    )

if __name__ =="__main__":
    app()
