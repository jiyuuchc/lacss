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

def pad_to(x, multiple=256):
    x = np.asarray(x)
    s = x.shape[0]
    ns = ((s - 1) // multiple + 1) * multiple
    padding = ns - s
    return np.pad(x, [[0,padding],[0,0]], constant_values=-1.0)

def get_mask(label, mask_pred):
    label = np.asarray(label).astype(int)
    mask_pred = np.asarray(mask_pred).astype(bool)
    keep = []
    for k, r in enumerate(regionprops(label)):
        y0, x0, y1, x1 = r['bbox']
        patch = mask_pred[y0:y1, x0:x1]
        if r['area'] < np.count_nonzero(r['image'] & patch) * 2:
            keep.append(k)
    return np.isin(label, keep)

@jax.jit
def get_mask_pred(model, img, loc):
    def _to_mask(img, pred):
        label = jnp.zeros(img.shape[:-1])
        label = label.at[pred['instance_yc'], pred['instance_xc']].max(pred['instance_output'])
        return label
    pred = model.train()(img, loc)
    mask = jax.vmap(_to_mask)(img, pred)
    mask = mask > 0.5
    return mask

def train_data_gen_fn(teacher, data_path):
    # platform_names = ['codex', 'cycif', 'imc', 'mibi', 'mxif', 'vectra']
    # tissue_names = ['breast', 'gi', 'immune', 'lung', 'pancreas', 'skin']
    # platforms = np.asarray([platform_names.index(p) for p in np.load(join(data_path, 'platform_list.npy'))])
    # tissues = np.asarray([tissue_names.index(t) for t in np.load(join(data_path, 'tissue_list.npy'))])

    X = np.load(join(data_path, 'train', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'train', 'y.npy'), mmap_mode='r+')
    L = np.load(join(data_path, 'train', 'segments.npy'), mmap_mode='r+')
    for x, y, l in zip(X, Y, L):
        img = x.astype('float32')
        label_in_ch0 = np.argmax(np.count_nonzero(y, axis=(0,1))) == 0
        y = y[..., 0] if label_in_ch0 else y[..., 1]
        locs = [prop['centroid'] for prop in regionprops(y)]

        mask_pred = get_mask_pred(
            teacher, 
            jnp.expand_dims(jnp.asarray(img), 0), 
            jnp.expand_dims(jnp.asarray(pad_to(locs)), 0),
        )
        binary_mask = get_mask(l, mask_pred[0])

        data = dict(
            image = img,
            locations =  locs,
            binary_mask = binary_mask,
        )
        data = lacss.data.parse_train_data_func(data, size_jitter=[0.8, 1.2])
        data = jax.tree_map(lambda v: jnp.asarray(v)[None,...], data)

        yield dict(image = data['image'], gt_locations = data['locations']), dict(binary_mask=data['binary_mask'])

def cb_fn(epoch, logs, model):
    model = model.eval()
    loiAP = lacss.metrics.MeanAP([0.1, 0.2, 0.5, 1.0])
    X = np.load(join(args.datapath, 'val', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(args.datapath, 'val', 'y.npy'), mmap_mode='r+')
    for x, y in tqdm(zip(X, Y)):
        img = jnp.expand_dims(jnp.asarray(x), 0).astype('float32')
        label_in_ch0 = np.argmax(np.count_nonzero(y, axis=(0,1))) == 0
        y = y[..., 0] if label_in_ch0 else y[..., 1]
        gt_locs = np.asarray([prop['centroid'] for prop in regionprops(y)])

        preds = model.predict_on_batch(img)
        pred = jax.tree_map(lambda v: v[0], preds) #unbatch

        scores = np.asarray(y['pred_scores'])
        mask = scores > 0
        pred_locs = np.asarray(pred['pred_locations'])[mask]
        dist2 = ((pred_locs[:,None,:] - gt_locs) ** 2).sum(axis=-1)
        sm = 1.0 / np.sqrt(dist2)
        loiAP.update_state(sm, scores[mask])

    loi_aps = loiAP.result()
    logs.update(dict(
        val_loi10_ap=loi_aps[0], 
        val_loi5_ap=loi_aps[1], 
        val_loi2_ap=loi_aps[2],
        val_loi1_ap=loi_aps[3],
        ))

def run_training():
    teacher = eg.load(args.cp)
    student = eg.Model(
        module = teacher.module.copy(),
        loss = [
            lacss.losses.DetectionLoss(),
            lacss.losses.LocalizationLoss(),
            lacss.losses.InstanceLoss(),
            lacss.losses.InstanceEdgeLoss(),
        ],
        optimizer = optax.adamw(0.001),
        seed = args.seed,
    )

    print(json.dumps(teacher.module.get_config(), indent=2))

    for epoch in range(15):
        print(f'epoch : {epoch}')
        teacher.module.merge(student.module.parameters(), inplace=True)
        student.fit(
            inputs=train_data_gen_fn(teacher, args.datapath), 
            epochs=1, 
            steps_per_epoch=2601,
            verbose = args.verbose,
            callbacks = [
                eg.callbacks.ModelCheckpoint(path=join(args.logpath, 'chkpt-{epoch:02d}')),
                eg.callbacks.LambdaCallback(on_epoch_end = partial(cb_fn, model=student)),
            ]
        )

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Train livecell model')
    parser.add_argument('datapath', type=str, help='Data dir of tfrecord files')
    parser.add_argument('logpath', type=str, help='Log dir for storing results')
    parser.add_argument('cp', type=str, default='', help='checkpoint name')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--verbose', type=int, default=2, help='output verbosity')

    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    try:
        os.makedirs(args.logpath)
    except:
        pass

    run_training()
