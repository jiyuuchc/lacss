#!/usr/bin/env python

from logging.config import valid_ident
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
from functools import partial
from os.path import join
from skimage.measure import regionprops

import numpy as np
import optax
import elegy as eg
import jax
jnp = jax.numpy

import lacss

import typer
app = typer.Typer(pretty_exceptions_enable=False)

# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

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

        data = dict(image = data['image'], gt_locations = pad_to(data['locations'].numpy())), dict(binary_mask=data['binary_mask'])        
        data = jax.tree_map(lambda v: jnp.asarray(v)[None,...], data)
        
        yield data

def cb_fn(epoch, logs, model, ds, datapath):
    model = model.eval()
    metrics = eg.metrics.Metrics([
        lacss.metrics.LoiAP([0.1, 0.2, 0.5, 1.0]),
    ])
    X = np.load(join(datapath, 'val', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(datapath, 'val', 'y.npy'), mmap_mode='r+')
    for x, y in zip(X, Y):
        img = jnp.expand_dims(jnp.asarray(x), 0).astype('float32')
        label_in_ch0 = np.argmax(np.count_nonzero(y, axis=(0,1))) == 0
        y = y[..., 0] if label_in_ch0 else y[..., 1]
        gt_locs = jnp.asarray([prop['centroid'] for prop in regionprops(y)])
        gt_locs = jnp.expand_dims(gt_locs, 0)

        preds = model.predict_on_batch(img)
        metrics.update(preds=preds, gt_locations = gt_locs)
        
    logs.update(metrics.compute())

@app.command()
def run_training(
    checkpoint:str, 
    datapath:str, 
    logpath:str='./', 
    verbose:int=2, 
    seed:int=42
):

    teacher = eg.load(checkpoint)
    student = eg.Model(
        module = teacher.module.copy(),
        loss = [
            lacss.losses.DetectionLoss(),
            lacss.losses.LocalizationLoss(),
            lacss.losses.InstanceLoss(),
            lacss.losses.InstanceEdgeLoss(),
        ],
        optimizer = optax.adamw(0.001),
        seed=seed,
    )

    print(json.dumps(teacher.module.get_config(), indent=2))

    for epoch in range(15):
        print(f'epoch : {epoch}')
        teacher.module.merge(student.module.parameters(), inplace=True)
        student.fit(
            inputs=train_data_gen_fn(teacher, datapath), 
            epochs=1, 
            steps_per_epoch=2601,
            verbose = verbose,
            callbacks = [
                eg.callbacks.ModelCheckpoint(path=join(logpath, 'chkpt-{epoch:02d}')),
                eg.callbacks.LambdaCallback(on_epoch_end = partial(cb_fn, model=student, datapath=datapath)),
            ]
        )

if __name__ == "__main__":
    app()
