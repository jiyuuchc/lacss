#!/usr/bin/env python

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

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
from tqdm import tqdm

import lacss

try:
    from . import data
except:
    import data

import typer

app = typer.Typer(pretty_exceptions_enable=False)


def get_model(cmd, config, seed):

    losses = [
        lacss.losses.detection_loss,
        lacss.losses.localization_loss,
        lacss.losses.supervised_instance_loss,
    ]

    if cmd == "resume":
        cp = lacss.train.Trainer.from_checkpoint(config)
        assert isinstance(cp, lacss.train.Trainer)
        trainer = cp
        logging.info(f"Loaded checkpoint {config}")

    elif cmd == "transfer":

        from lacss.deploy import load_from_pretrained

        module, params = load_from_pretrained(config)
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
        model = lacss.modules.Lacss.from_config(model_cfg)

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
    datapath: str = "../../livecell_dataset",
    logpath: str = "",
    celltype: int = -1,
    seed: int = 42,
    batchsize: int = 1,
    n_epochs: int = 10,
    steps_per_epoch: int = 3500 * 5,
    init_epoch: int = 0,
    lr: float = 0.002,
    coco: bool = False,
    n_buckets: int = 4,
):
    tf.random.set_seed(seed)

    if logpath == "":
        import time

        logpath = join(".", time.strftime("%Y%m%d-%H%M%S"))
    try:
        os.makedirs(logpath)
    except:
        pass

    logging.info(f"Logging to {logpath}")

    train_gen = data.train_data(
        datapath,
        n_buckets,
        batchsize,
        supervised=True,
        cell_type=celltype,
        coco=coco,
    )
    val_gen = data.val_data(
        datapath,
        supervised=True,
        cell_type=celltype,
        coco=coco,
        batch=False,
    )  # unbatched val data

    trainer = get_model(cmd, config, seed)

    # initialize before calling asdict because the model may be bound
    steps_per_epoch = steps_per_epoch // batchsize
    if not trainer.initialized:
        schedules = [
            optax.cosine_decay_schedule(lr * batchsize, steps_per_epoch)
        ] * n_epochs
        boundaries = [steps_per_epoch * i for i in range(1, n_epochs)]
        schedule = optax.join_schedules(schedules, boundaries)

        trainer.initialize(train_gen, optax.adam(schedule))

    logging.info(
        json.dumps(dataclasses.asdict(trainer.model), indent=2, sort_keys=True)
    )

    train_iter = trainer.train(train_gen, rng_cols=["droppath"], training=True)
    for epoch in range(init_epoch, n_epochs):

        trainer.reset()
        print(f"epoch - {epoch+1}")

        for _ in tqdm(range(steps_per_epoch)):
            logs = next(train_iter)

        print(", ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))

        trainer.checkpoint(join(logpath, f"cp-{epoch}"))

        val_metrics = [
            lacss.metrics.LoiAP([0.2, 0.5, 1.0]),
            lacss.metrics.BoxAP([0.5, 0.75]),
        ]

        var_logs = trainer.test_and_compute(
            val_gen, metrics=val_metrics, strategy=lacss.train.JIT
        )
        for k, v in var_logs.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    # import warnings
    # warnings.simplefilter(action='ignore', category=FutureWarning)

    logging.basicConfig(level=logging.INFO)

    app()
