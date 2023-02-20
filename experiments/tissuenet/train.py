#!/usr/bin/env python

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

def pad_to(x, multiple=256):
    x = np.asarray(x)
    s = x.shape[0]
    ns = ((s - 1) // multiple + 1) * multiple
    padding = ns - s
    return np.pad(x, [[0,padding],[0,0]], constant_values=-1.0)

def train_dataset(data_path, supervised):
    X = np.load(join(data_path, 'train', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'train', 'y.npy'), mmap_mode='r+')
    while (True):
        S = np.arange(len(X))
        np.random.shuffle(S)
        for k in S:
            image = X[k].astype('float32')
            label = Y[k][..., 0]
            locs = np.asarray([prop['centroid'] for prop in regionprops(label)])

            image = tf.image.random_contrast(image, 0.6, 1.4)
            image = tf.image.random_brightness(image, 0.3)
            if tf.random.uniform([]) >= .5:
                image = tf.image.transpose(image)
                locs = locs[..., ::-1]
                label = tf.transpose(label)

            if supervised:
                data = (
                    dict(image = image, gt_locations = pad_to(locs),),
                    dict(mask_labels = label),
                )
            else:
                augmented = lacss.data.parse_train_data_func(
                    dict(
                        image = image,
                        locations = locs,
                        binary_mask = (label > 0).astype('float32')
                    ),
                    size_jitter=[0.85,1.15],
                )
                data = (
                    dict(image = augmented['image'].numpy(), gt_locations = pad_to(augmented['locations'].numpy())),
                    dict(binary_mask = augmented['binary_mask'].numpy()),
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

def get_schedule(schedule, batchsize=1):
    if schedule == 1:
        n_epochs = 10
        steps_per_epoch = 10000
        schedules = [optax.cosine_decay_schedule(0.002 * batchsize, steps_per_epoch)] * n_epochs
        boundaries = [steps_per_epoch * i for i in range(1, n_epochs)]
        lr = optax.join_schedules(schedules, boundaries)
    else:
        raise ValueError()

    return n_epochs, steps_per_epoch, lr

@app.command()
def run_training(
    cmd: str,
    config: str,
    datapath: str = '../tissue_net/',
    logpath: str = './',
    seed: int = 42,
    supervised: bool = True,
    schedule: int = 1,
):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    try:
        os.makedirs(logpath)
    except:
        pass

    ds_train = partial(train_dataset, data_path = datapath, supervised=supervised)
    ds_val = partial(val_dataset, data_path = datapath)
    n_epochs, steps_per_epoch, lr = get_schedule(schedule)

    if cmd == 'resume':
        trainer = lacss.train.Trainer.from_checkpoint(config)
        trainer._strategy = lacss.train.JIT
        print(f'Loaded checkpoint {config}')
        try:
            init_epoch = int(config.split('-')[-1])
        except:
            init_epoch = 0
    else:
        if cmd == 'transfer':
            with open(config, 'rb') as f:
                model = pickle.load(f)
            if not isinstance(model, tx.Module):
                if 'model' in vars(model):
                    model = model.model
                elif 'module' in vars(model):
                    model = model.module
            assert isinstance(model, tx.Module)
            model.freeze(False, inplace=True)
            model.detector._config_dict['test_max_output'] = 1024
            
        elif cmd == 'config':
            with open(config) as f:
                model_cfg = json.load(f)
            model = lacss.modules.Lacss.from_config(model_cfg)
        else:
            raise ValueError('Cmd must be one of the "resume", "transfer" or "config"')
        
        if supervised:
            losses = [
                    lacss.losses.DetectionLoss(),
                    lacss.losses.LocalizationLoss(),
                    lacss.losses.SupervisedInstanceLoss(),
                ]
        else:
            losses = [
                    lacss.losses.DetectionLoss(),
                    lacss.losses.LocalizationLoss(),
                    lacss.losses.InstanceLoss(),
                    lacss.losses.InstanceEdgeLoss(),
                ]
        trainer = lacss.train.Trainer(
            model = model,
            optimizer = optax.adamw(lr),
            losses = losses,
            seed = seed,
        )
        init_epoch = 0

    print(json.dumps(trainer.model.get_config(), indent=2, sort_keys=True))

    test_metrics = tx.metrics.Metrics([
        lacss.metrics.LoiAP([0.2, 0.5, 1.0]),
        lacss.metrics.BoxAP([0.5, 0.75])
    ])

    epoch = init_epoch + 1
    print(f'epoch - {epoch}')
    for steps, logs in enumerate(tqdm(trainer(ds_train))):
        if (steps + 1) % steps_per_epoch == 0:
            print(", ".join([f'{k}:{v:.4f}' for k,v in logs.items()]))

            trainer.checkpoint(join(logpath, f'cp-{epoch}'))

            val_logs = trainer.test_and_compute(ds_val, test_metrics, lacss.train.Core)
            for k,v in val_logs.items():
                print(f'{k}: {v}')

            epoch += 1
            if epoch == n_epochs:
                break
            
            print(f'epoch - {epoch}', flush=True)
            trainer.reset_metrics()

if __name__ =="__main__":
    app()
