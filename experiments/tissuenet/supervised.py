#!/usr/bin/env python

import dataclasses
import json
import logging
import os
from functools import partial
from logging.config import valid_ident
from os.path import join

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import lacss

try:
    from . import data
except:
    import data

import typer

app = typer.Typer(pretty_exceptions_enable=False)

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


def get_model(cmd, config, seed):

    losses = [
        lacss.losses.LPNLoss(),
        lacss.losses.SupervisedInstanceLoss(),
    ]

    if cmd == "resume":
        cp = lacss.train.Trainer.from_checkpoint(config)
        assert isinstance(cp, lacss.train.Trainer)
        trainer = cp
        logging.info(f"Loaded checkpoint {config}")

    elif cmd == "transfer":

        from lacss.deploy import _from_pretrained

        module, params = _from_pretrained(config)
        model = module.bind(dict(params=params))

        trainer = lacss.train.Trainer(
            model=model,
            losses=losses,
            seed=seed,
            strategy=lacss.train.strategy.VMapped,
        )

    elif cmd == "config":
        with open(config) as f:
            model_cfg = json.load(f)
        model = lacss.modules.Lacss(**model_cfg)

        trainer = lacss.train.Trainer(
            model=model,
            losses=losses,
            seed=seed,
            strategy=lacss.train.strategy.VMapped,
        )

    else:
        raise ValueError('Must be one of "resume", "transfer" or "config"')

    return trainer


@app.command()
def run_training(
    cmd: str,
    config: str,
    datapath: str = "../../../tissue_net/",
    logpath: str = "./",
    seed: int = 42,
    n_buckets: int = 4,
    batchsize: int = 1,
    init_epoch: int = 0,
    n_epochs: int = 10,
    steps_per_epoch: int = 10000,
    lr: float = 0.002,
    nucleus: bool = False,
):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    steps_per_epoch = steps_per_epoch // batchsize
    ch = 1 if nucleus else 0

    try:
        os.makedirs(logpath)
    except:
        pass

    train_gen = data.train_data_supervised(datapath, n_buckets, batchsize, ch=ch)
    val_gen = data.val_data_supervised(datapath, n_buckets, 4, ch=ch)

    trainer = get_model(cmd, config, seed)

    if not trainer.initialized:
        schedules = [
            optax.cosine_decay_schedule(lr * batchsize, steps_per_epoch)
        ] * n_epochs
        boundaries = [steps_per_epoch * i for i in range(1, n_epochs)]
        schedule = optax.join_schedules(schedules, boundaries)

        trainer.initialize(train_gen, optax.adam(schedule))
        # trainer.initialize(train_gen, optax.adam(lr))

    logging.info(
        json.dumps(dataclasses.asdict(trainer.model), indent=2, sort_keys=True)
    )

    epoch = init_epoch
    for steps, logs in enumerate(
        trainer.train(train_gen, rng_cols=["droppath"], training=True)
    ):

        if (steps + 1) % steps_per_epoch == 0:
            epoch += 1
            print(f"epoch - {epoch}")
            print(", ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))

            trainer.checkpoint(join(logpath, f"cp-{epoch}"))
            trainer.reset()

            val_metrics = [
                lacss.metrics.LoiAP([0.2, 0.5, 1.0]),
                lacss.metrics.BoxAP([0.5, 0.75]),
            ]
            var_logs = trainer.test_and_compute(val_gen, metrics=val_metrics)
            for k, v in var_logs.items():
                print(f"{k}: {v}")

            if epoch >= n_epochs:
                break


if __name__ == "__main__":
    # import warnings
    # warnings.simplefilter(action='ignore', category=FutureWarning)

    logging.basicConfig(level=logging.INFO)

    app()
