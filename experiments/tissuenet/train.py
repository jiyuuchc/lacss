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

from data import *
from util import *

def get_schedule(schedule, batchsize=1):
    if schedule == 1:
        n_epochs = 20
        steps_per_epoch = 10000
        schedules = [optax.cosine_decay_schedule(0.002 * batchsize, steps_per_epoch)] * n_epochs
        boundaries = [steps_per_epoch * i for i in range(1, n_epochs)]
        lr = optax.join_schedules(schedules, boundaries)
    elif schedule == 2:
        n_epochs = 30
        steps_per_epoch = 2601
        lr = 0.001
    else:
        raise ValueError('Invalud schedule')

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

            # limit max output due to the val images are small
            if model.detector._config_dict['test_max_output'] > 1024:
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
    steps = 0
    file_writer = tf.summary.create_file_writer(join(logpath, 'train'))
    print(f'epoch - {epoch}')
    while (epoch <= n_epochs):
        for logs in tqdm(trainer(ds_train)):
            steps += 1
            if steps  % steps_per_epoch == 0:
                print(", ".join([f'{k}:{v:.4f}' for k,v in logs.items()]))

                trainer.checkpoint(join(logpath, f'cp-{epoch}'))

                preds = sample_prediction(datapath, trainer.model.train())
                label = to_label(preds)[...,None].astype(float)
                label = label / label.max()
                with file_writer.as_default():
                    tf.summary.image("label_pred", label, step=epoch, max_outputs=8)

                val_logs = trainer.test_and_compute(ds_val, test_metrics, lacss.train.Core)
                for k,v in val_logs.items():
                    print(f'{k}: {v}')

                epoch += 1
                if epoch > n_epochs:
                    break
                
                print(f'epoch - {epoch}', flush=True)
                trainer.reset_metrics()

if __name__ =="__main__":
    app()
