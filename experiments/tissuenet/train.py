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
from skimage.measure import regionprops
from tqdm import tqdm

jnp = jax.numpy

import lacss

tf.config.set_visible_devices([], 'GPU')

# global defaults
batch_size = 1
learning_rate = 2e-3 * batch_size
training_epochs = 16
steps_per_epoch = 3500 * 5 // batch_size

def tissue_net_gen_fn(data_path):
    X = np.load(join(data_path, 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'y.npy'), mmap_mode='r+')
    platforms = np.load(join(data_path, 'platform_list.npy'))
    tissues = np.load(join(data_path, 'tissue_list.npy'))
    for x, y, pf, t in zip(X, Y, platforms, tissues):
        img = x.astype('float32')
        img = np.pad(img, [[0,0],[0,0],[0,1]]) # make 3-ch
        label_in_ch0 = np.argmax(np.count_nonzero(y, axis=(0,1))) == 0
        y = y[..., 0] if label_in_ch0 else y[..., 1]
        binary_mask = (y > 0).astype('float32')
        locs = [prop['centroid'] for prop in regionprops(y)]
        bboxes = []
        for prop in regionprops(y):
            bboxes.append(prop['bbox'])
        
        bboxes = np.array(bboxes, dtype='float32')

        yield {
            'image': img,
            'locations': locs,
            'binary_mask': binary_mask,
            'bboxes': bboxes,
            'mask_labels': y,
            'platform': pf,
            'tissue': t,
        }

def train_parser(x):
    platform_lookup = dict(
        codex = 0,
        cyclif = 1,
        img = 2,
        mibi = 3,
        mxif = 4,
        vectra = 5,
    )
    pf = tf.constant(platform_lookup[x['platform']], tf.float32)
    x = lacss.data.parse_train_data_func(x, size_jitter=(0.85, 1.1), target_height=512, target_width=512)
    if tf.random.uniform([]) >=0.5:
        x['image'] = tf.image.transpose(x['image'])
        x['binary_mask'] = tf.transpose(x['binary_mask'])
        x['locations'] = x['locations'][:,::-1]
    
    x_data = dict(
        image = tf.pad(x['image'], [[0,0],[0,0],[0,1]]),
        gt_locations = x['locations'],
    )
    y_data = dict(
        group_num = pf,
        binary_mask = tf.cast(x['binary_mask'], tf.float32),
    )
    return x_data, y_data

def prepare_data():
    output_signiture =  {
            'image': tf.TensorSpec([None, None, 3], dtype=tf.float32),
            'locations': tf.TensorSpec([None, 2], dtype=tf.float32),
            'binary_mask': tf.TensorSpec([None, None], dtype=tf.float32),
            'bboxes': tf.TensorSpec([None, 4], dtype=tf.float32),
            'mask_labels': tf.TensorSpec([None, None], tf.float32),
            'platform': tf.TensorSpec([], tf.string),
            'tissue': tf.TensorSpec([], tf.string),
        }

    ds_train = tf.data.Dataset.from_generator(
        lambda: tissue_net_gen_fn(join(args.datapath, 'train')),
        output_signature = output_signiture
    ).map(train_parser)

    ds_train = ds_train.bucket_by_sequence_length(
        element_length_func = lambda x, y: tf.shape(x['gt_locations'])[0],
        bucket_boundaries = list(np.arange(1, 32) * 128 + 1),
        bucket_batch_sizes = (batch_size,) * 32,
        padding_values = -1.0,
        pad_to_bucket_boundary = True,
    )

    ds_val = tf.data.Dataset.from_generator(
        lambda: tissue_net_gen_fn(join(args.datapath, 'val')),
        output_signature = output_signiture
    )

    return ds_train, ds_val

def cb_fn(epoch, logs, model, ds):
    model = model.eval()
    loiAP = lacss.metrics.MeanAP([0.1, 0.2, 0.5, 1.0])
    boxAP = lacss.metrics.MeanAP([0.5, 0.75])
    for x in tqdm(ds):
        image = jnp.array(x['image'])
        y = model.predict_on_batch(image[None, ...])
        y = jax.tree_map(lambda v: v[0], y) #unbatch

        scores = np.array(y['pred_scores'])
        mask = scores > 0

        gt_locs = x['locations'].numpy()
        pred_locs = np.array(y['pred_locations'])
        dist2 = ((pred_locs[:,None,:] - gt_locs) ** 2).sum(axis=-1)
        sm = 1.0 / np.sqrt(dist2)
        loiAP.update_state(sm, scores[mask])

        gt_box = x['bboxes'].numpy()
        pred_box = lacss.ops.bboxes_of_patches(y)
        sm = lacss.ops.box_iou_similarity(pred_box, jnp.array(gt_box))
        boxAP.update_state(np.array(sm)[mask], scores[mask])

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

def run_training():
    ds_train, ds_val = prepare_data()
    print(ds_train.element_spec)

    if args.resume != "":
        model = eg.model.model_base.load(args.resume)
        print(f'Loaded checkpoint {args.resume}')
        init_epoch = int(args.resume.split('-')[-1])
    else:
        if args.config != "":
            with open(args.config) as f:
                model_cfg = json.load(f)
        else:
            model_cfg = dict(
                detector = dict(
                    train_pre_nms_topk = 2048,
                    train_max_output = 1024,
                    train_min_score = 0.4,
                    test_pre_nms_topk = -1,
                    test_max_output = 1024,
                    test_min_score = 0.2,
                ),
                auxnet = dict(
                    n_groups=6,
                )
            )
        module = lacss.modules.Lacss.from_config(model_cfg)

        schedules = [optax.cosine_decay_schedule(learning_rate, steps_per_epoch)] * training_epochs
        boundaries = [steps_per_epoch * i for i in range(1, training_epochs)]
        schedule = optax.join_schedules(schedules, boundaries)
        optimizer = optax.adamw(schedule)

        loss = [
            lacss.losses.DetectionLoss(),
            lacss.losses.LocalizationLoss(),
            loss.append(lacss.losses.AuxnetLoss())
            loss.append(lacss.losses.InstanceOverlapLoss())
            loss.append(lacss.losses.ImageLevelSegmentationLoss())
        ]

        model = eg.Model(
            module = module,     
            optimizer = optimizer,
            seed = args.seed,
            loss = loss,
        )
        init_epoch = 0

    with open(join(args.logpath, 'config.json'), 'w') as f:
        json.dump(model.module.get_config(), f)

    model.fit(
        inputs=ds_train, 
        epochs=training_epochs, 
        steps_per_epoch=steps_per_epoch, 
        initial_epoch=init_epoch,
        verbose = args.verbose,
        callbacks = [
            eg.callbacks.TensorBoard(args.logpath),
            eg.callbacks.ModelCheckpoint(path=join(args.logpath, 'chkpt-{epoch:02d}')),
            eg.callbacks.LambdaCallback(on_epoch_end = partial(cb_fn, model=model, ds=ds_val)),
        ]
    )

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Train livecell model')
    parser.add_argument('datapath', type=str, help='Data dir of tfrecord files')
    parser.add_argument('logpath', type=str, help='Log dir for storing results')
    parser.add_argument('--config', type=str, default='', help='path to the model config file')   
    # parser.add_argument('--supervised', type=bool, default=False, help='Whether train superversed')
    parser.add_argument('--resume', type=str, default="", help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--verbose', type=int, default=2, help='output verbosity')

    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    try:
        os.makedirs(args.logpath)
    except:
        pass

    run_training()
