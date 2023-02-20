from logging.config import valid_ident
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
from functools import partial
import os
from os.path import join
import pickle

from tqdm import tqdm
from skimage.measure import regionprops
import numpy as np
import optax
import treex as tx
import jax
jnp = jax.numpy

import lacss

import typer
app = typer.Typer(pretty_exceptions_enable=False)

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def _to_mask(img, pred):
    label = jnp.zeros(img.shape[:-1])
    label = label.at[pred['instance_yc'], pred['instance_xc']].max(pred['instance_output'])
    return label

class SizeLoss(tx.Loss):
    def __init__(self, a, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def call(self, preds):
        valid_locs = (preds['instance_yc'] >= 0) & (preds['instance_yc'] < 512) & (preds['instance_xc'] >= 0) & (preds['instance_xc'] < 512)
        areas = jnp.sum(preds['instance_output'], axis=(2,3), where=valid_locs, keepdims=True) / 96 / 96
        loss =  self.a / (jnp.sqrt(areas)+1e-8)
        loss = jnp.sum(loss, axis=1, where=preds['instance_mask']) / (jnp.count_nonzero(preds['instance_mask'], axis=(1,2,3)) + 1e-8)
        return loss

class LOIInstanceLoss(lacss.losses.InstanceLoss):
    def call(self, preds):
        mask_pred = jax.lax.stop_gradient((preds['fg_pred'] >= .5).astype("float32"))
        return super().call(preds=preds, binary_mask = mask_pred.squeeze(-1))

class LOIMaskPredLoss(tx.Loss):
    def call(self, inputs, preds):
        mask_from_instances = jax.lax.stop_gradient((jax.vmap(_to_mask)(inputs['image'], preds) >= 0.5)).astype(int)
        mask_from_instances = mask_from_instances[..., None]
        p_t = mask_from_instances * preds['fg_pred'] + (1 - mask_from_instances) * (1.0 - preds['fg_pred'])
        bce = - jnp.log(jnp.clip(p_t, 1e-9, 1.0)).mean(axis=(1,2,3))
        return bce

class ForegroundPredict(tx.Module):
    @tx.compact
    def __call__(self, image):
        x = image
        x = tx.Conv(24, (3,3), strides=(2,2), use_bias=False)(x)
        x = tx.LayerNorm(use_scale=False)(x)
        x = jax.nn.relu(x)

        x = tx.Conv(64, (3,3), use_bias=False)(x)
        x = tx.LayerNorm(use_scale=False)(x)
        x = jax.nn.relu(x)

        x = tx.Conv(64, (3,3), use_bias=False)(x)
        x = tx.LayerNorm(use_scale=False)(x)
        x = jax.nn.relu(x)

        x = tx.ConvTranspose(1, (2,2), strides=(2,2))(x)
        x = jax.nn.sigmoid(x)

        return x

class LOILacss(tx.Module):
    def __init__(self, lacss_module, **kwargs):
        super().__init__(**kwargs)
        self.lacss_module = lacss_module
    
    @tx.compact
    def __call__(self, image, gt_locations=None):
        preds = self.lacss_module(image, gt_locations)
        preds.update(dict(
            fg_pred = ForegroundPredict()(image)
        ))
        return preds

def pad_to(x, multiple=256):
    x = np.asarray(x)
    s = x.shape[0]
    ns = ((s - 1) // multiple + 1) * multiple
    padding = ns - s
    return np.pad(x, [[0,padding],[0,0]], constant_values=-1.0)

def train_dataset(data_path):
    X = np.load(join(data_path, 'train', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'train', 'y.npy'), mmap_mode='r+')
    S = np.arange(len(X))
    np.random.shuffle(S)
    for k in S:
        image = X[k].astype('float32')
        label = Y[k][..., 0]
        locs = np.asarray([prop['centroid'] for prop in regionprops(label)])
        mask = np.zeros(image.shape[:-1])

        # tf pipeline
        image = tf.image.random_contrast(image, 0.6, 1.4)
        image = tf.image.random_brightness(image, 0.3)
        if tf.random.uniform([]) >= .5:
            image = tf.image.transpose(image)
            locs = locs[..., ::-1]        
        augmented = lacss.data.parse_train_data_func(dict(
            image = image,
            locations = locs,
            binary_mask = mask,
        ), size_jitter=[.85, 1.15])

        # back to numpy
        image = augmented['image'].numpy()
        locs = augmented['locations'].numpy()

        data = (
            dict(image = image,gt_locations = pad_to(locs),),
            dict(mask_labels = label),
        )
        data = jax.tree_map(lambda v: jnp.asarray(v)[None,...], data)        

        yield data

def val_dataset(data_path):
    X = np.load(join(data_path, 'val', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'val', 'y.npy'), mmap_mode='r+')
    for k in range(len(X)):
        image = X[k].astype('float32')
        label = Y[k][..., 0]
        props = regionprops(label)
        locs = np.asarray([prop['centroid'] for prop in props])
        bboxes = np.asarray([prop['bbox'] for prop in props])
        data = (
            dict(image = image),
            dict(gt_boxes = pad_to(bboxes), gt_locations = pad_to(locs)),
            )
        data = jax.tree_map(lambda v: jnp.asarray(v)[None,...], data)        

        yield data

@app.command()
def run_training(
    checkpoint:str,
    datapath:str = '../tissue_net/', 
    logpath:str='./', 
    n_epochs:int=30,
    init_epoch:int=0,
    seed:int=42
):
    with open(checkpoint, 'rb') as f:
        cp = pickle.load(f)
    if isinstance(cp, tx.Module):
        lacss_module = cp
        lacss_module.detector._config_dict['test_max_output'] = 1024

        loi_model = LOILacss(lacss_module)
        trainer = lacss.train.Trainer(
            model = loi_model,
            losses = [
                lacss.losses.DetectionLoss(),
                lacss.losses.LocalizationLoss(),
                SizeLoss(1e-3),
                LOIMaskPredLoss(),
                LOIInstanceLoss(),
            ],
            optimizer = optax.adamw(0.001),
            seed=seed,
        )
    else:
        trainer = cp

    assert isinstance(trainer, lacss.train.Trainer)

    ds_train = partial(train_dataset, data_path = datapath)
    ds_val = partial(val_dataset, data_path = datapath)
    test_metrics = tx.metrics.Metrics([
        lacss.metrics.LoiAP([0.2, 0.5, 1.0]),
        lacss.metrics.BoxAP([0.5, 0.75])
    ])
    for epoch in range(init_epoch+1, n_epochs+1):
        print(f'epoch-{epoch}')
        for logs in tqdm(trainer(ds_train)):
            pass
        print(", ".join([f'{k}:{v:.4f}' for k,v in logs.items()]))

        trainer.checkpoint(join(logpath, f'cp-{epoch}'))

        val_logs = trainer.test_and_compute(ds_val, test_metrics, lacss.train.Core)
        for k,v in val_logs.items():
            print(f'{k}: {v}')

if __name__ == "__main__":
    app()
