#!/usr/bin/env python

from logging.config import valid_ident
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import json
from functools import partial
import argparse
from os.path import join
import numpy as np
import jax
import optax
import elegy as eg
import tensorflow as tf

jnp = jax.numpy

import lacss
try:
    from . import data
except:
    import data

tf.config.set_visible_devices([], 'GPU')

def pad_to(x, multiple=512):
    s = x.shape[0]
    ns = ((s - 1) // multiple + 1) * multiple
    padding = ns - s
    return np.pad(x, [[0,padding],[0,0]]), s

def cb_fn(epoch, logs, model, ds):
    # if args.schedule==1 and epoch < 10: 
    #     return
    model = model.eval()
    loiAP = lacss.metrics.MeanAP([0.1, 0.2, 0.5, 1.0])
    boxAP = lacss.metrics.MeanAP([0.5, 0.75])
    for x in ds:
        y = model.predict_on_batch(jnp.array(x['image']))
        y = jax.tree_map(lambda v: v[0], y)

        scores = np.array(y['pred_scores'])
        mask = scores > 0

        gt_locs = x['locations'].numpy()
        pred_locs = np.array(y['pred_locations'])
        sm = ((pred_locs[mask, None, :] - gt_locs) ** 2).sum(axis=-1)
        sm = 1.0 / np.sqrt(sm)
        loiAP.update_state(sm, scores[mask])

        gt_box, sz = pad_to(x['bboxes'].numpy()[0])
        pred_box = lacss.ops.bboxes_of_patches(y)
        sm = lacss.ops.box_iou_similarity(pred_box, jnp.array(gt_box))
        boxAP.update_state(np.array(sm)[mask, :sz], scores[mask])

    loi_aps = loiAP.result()
    logs.update(dict(
        val_loi10_ap=loi_aps[0], 
        val_loi5_ap=loi_aps[1], 
        val_loi2_ap=loi_aps[2],
        val_loi1_ap=loi_aps[3],
        ))

    box_aps = boxAP.result()
    logs.update(dict(
        val_box_ap_50=box_aps[0],
        val_box_ap_75=box_aps[1],
        ))

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
    
    if tf.random.uniform([]) >= .5:
        image = tf.image.transpose(image)
        gt_locations = gt_locations[..., ::-1]
        image_mask = tf.transpose(image_mask)

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
    inputs = lacss.data.parse_test_data_func(inputs)
    return inputs

def prepare_data(n_buckets = 16):
    ds_train = data.livecell_dataset_from_tfrecord(join(args.datapath, 'train.tfrecord'))

    if args.celltype >= 0:
        ds_train = ds_train.filter(lambda x: x['cell_type'] == args.celltype)

    train_parser = train_parser_supervised if args.supervised else train_parser_semisupervised
    ds_train = ds_train.map(train_parser)
    ds_train = ds_train.filter(lambda x, _: tf.size(x['gt_locations'])>0).repeat()    
    ds_train = ds_train.bucket_by_sequence_length(
        element_length_func = lambda x, y: tf.shape(x['gt_locations'])[0],
        bucket_boundaries = list(np.arange(1, n_buckets) * (4096 // n_buckets) + 1),
        bucket_batch_sizes = (args.batchsize,) * n_buckets,
        padding_values = -1.0,
        pad_to_bucket_boundary = True,
    )

    ds_val = data.livecell_dataset_from_tfrecord(join(args.datapath, 'val.tfrecord'))
    ds_val = ds_val.map(val_parser).batch(1)

    return ds_train, ds_val

def get_schedule():
    if args.schedule == 1:
        n_epochs = 30
        steps_per_epoch = 3500
        optimizer = optax.adam(0.0005)
    elif args.schedule == 2:
        n_epochs = 10
        steps_per_epoch = 3500 * 5
        schedules = [optax.cosine_decay_schedule(0.002 * args.batchsize, steps_per_epoch)] * n_epochs
        boundaries = [steps_per_epoch * i for i in range(1, n_epochs)]
        schedule = optax.join_schedules(schedules, boundaries)
        optimizer = optax.adamw(schedule)
    else:
        raise ValueError()

    return n_epochs, steps_per_epoch, optimizer

def get_model(optimizer):
    if args.resume != "":
        model = eg.model.model_base.load(args.resume)
        print(f'Loaded checkpoint {args.resume}')
        try:
            init_epoch = int(args.resume.split('-')[-1])
        except:
            init_epoch = 0
        return model, init_epoch        

    if args.transfer != "":
        model = eg.model.model_base.load(args.transfer)
        module = model.module
        module.freeze(False, inplace=True)
    elif args.config !="":
        with open(args.config) as f:
            model_cfg = json.load(f)
        module = lacss.modules.Lacss.from_config(model_cfg)
    else:
        raise ValueError('Must specify at least one of the "--resume", "--transfer" or "--config"')

    loss = [
        lacss.losses.DetectionLoss(),
        lacss.losses.LocalizationLoss(),
    ]
    if not args.supervised:
        loss.append(lacss.losses.InstanceEdgeLoss())
        loss.append(lacss.losses.InstanceLoss())
    else:
        loss.append(lacss.losses.SupervisedInstanceLoss())
        pass

    model = eg.Model(
        module = module,     
        optimizer = optimizer,
        seed = args.seed,
        loss = loss,
    )
    init_epoch = 0

    return model, init_epoch

def run_training():
    ds_train, ds_val = prepare_data()
    print(ds_train.element_spec)

    n_epochs, steps_per_epoch, optimizer = get_schedule()
    model, init_epoch = get_model(optimizer)

    print(json.dumps(model.module.get_config(), indent=2, sort_keys=True))

    callbacks = [
        eg.callbacks.TensorBoard(args.logpath),
        eg.callbacks.ModelCheckpoint(path=join(args.logpath, 'chkpt-{epoch:02d}')),
        eg.callbacks.LambdaCallback(on_epoch_end = partial(cb_fn, model=model, ds=ds_val)),
    ]
    model.fit(
        inputs=ds_train, 
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=init_epoch,
        verbose = args.verbose,
        callbacks = callbacks,
    )

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Train livecell model')
    parser.add_argument('datapath', type=str, help='Data dir of tfrecord files')
    parser.add_argument('logpath', type=str, help='Log dir for storing results')
    parser.add_argument('--config', type=str, default="", help='path to the model config file')   
    parser.add_argument('--resume', type=str, default="", help='Resume from checkpoint')
    parser.add_argument('--transfer', type=str, default="", help='Transfer from previous model')
    parser.add_argument('--celltype', type=int, default=-1, help='Cell type 0-7')
    parser.add_argument('--supervised', type=bool, default=False, help='Whether train superversed')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--verbose', type=int, default=2, help='output verbosity')
    parser.add_argument('--schedule', type=int, default=1, choices=[1,2], help='trainging schedule')
    parser.add_argument('--batchsize', type=int, default=1, help='batch size')
 
    args = parser.parse_args()

    print(args)
    print()

    tf.random.set_seed(args.seed)

    try:
        os.makedirs(args.logpath)
    except:
        pass

    run_training()
