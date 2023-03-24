#!/usr/bin/env python

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import dataclasses
import itertools
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
import typer

import lacss

app = typer.Typer(pretty_exceptions_enable=False)


def remove_label(g):
    for inputs, label in g():
        yield inputs


@app.command()
def run_training(
    config: str = "",
    resume: str = "",
    datapath: str = "../../../livecell_dataset",
    logpath: str = "",
    cell_type: int = -1,
    seed: int = 42,
    batchsize: int = 1,
    n_epochs: int = 15,
    steps_per_epoch: int = 5000,
    init_epoch: int = 0,
    lr: float = 0.001,
    coco: bool = False,
    n_buckets: int = 4,
    size_loss: float = 2.0,
):
    import data

    tf.random.set_seed(seed)

    try:
        os.makedirs(logpath)
    except:
        pass

    train_data = data.train_data(datapath, n_buckets, batchsize, cell_type=cell_type)
    train_data = partial(remove_label, train_data)
    val_data = data.val_data(datapath, cell_type=cell_type)

    if len(config) > 0:

        with open(config) as f:
            lacss_cfg = json.load(f)

        cfg = dict(
            cfg=lacss_cfg,
            aux_edge_cfg=dict(n_groups=8),
            aux_fg_cfg=dict(
                n_groups=8,
                conv_spec=(16, 32, 64),
                # share_weights=True,
            ),
        )
        trainer = lacss.train.Trainer(
            model=lacss.modules.lacss.LacssWithHelper(**cfg),
            losses=[
                lacss.losses.LPNLoss(),
                lacss.losses.AuxEdgeLoss(),
                lacss.losses.WeaklySupervisedInstanceLoss(ignore_mask=True),
                lacss.losses.AuxSizeLoss(size_loss),
                lacss.losses.AuxSegLoss(offset_sigma=50.0, offset_scale=1.0),
            ],
            seed=seed,
            strategy=lacss.train.strategy.VMapped,
        )

    elif len(resume) > 0:

        trainer = lacss.train.Trainer.from_checkpoint(resume)

    else:

        raise ValueError("Must specify either config file or checkpoint")

    if not trainer.initialized:

        trainer.initialize(val_data, optax.adam(lr * batchsize))

    file_writer = tf.summary.create_file_writer(join(logpath, "train"))
    to_label = jax.vmap(lacss.ops.patches_to_label, in_axes=(0, None))

    epoch = init_epoch
    train_iter = trainer.train(
        train_data, rng_cols=["droppath", "augment"], training=True
    )

    while epoch < n_epochs:

        epoch += 1
        print(f"epoch - {epoch}")

        for _ in range(steps_per_epoch):

            logs = next(train_iter)

        print(", ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))

        trainer.checkpoint(join(logpath, f"cp-{epoch}"))
        trainer.reset()

        # val_metrics = [
        #     lacss.metrics.LoiAP([0.1, 0.2, 0.5, 1.0]),
        #     lacss.metrics.BoxAP([0.5, 0.75]),
        # ]
        # var_logs = trainer.test_and_compute(val_data, metrics=val_metrics)
        # for k, v in var_logs.items():
        #     print(f"{k}: {v}")

        for c in range(8) if cell_type == -1 else [cell_type]:

            data = itertools.dropwhile(lambda x: x[0]["category"] != c, val_data())
            inputs, label = next(data)
            preds = trainer(inputs)
            label = to_label(preds, inputs["image"].shape[1:3])
            output = jnp.concatenate(
                [
                    inputs["image"],
                    label[..., None] / label.max(),
                    jax.nn.sigmoid(preds["fg_pred"][..., None]),
                ],
                axis=2,
            )

            with file_writer.as_default():
                tf.summary.image(f"img_{c}", output, epoch)


if __name__ == "__main__":
    # import warnings
    # warnings.simplefilter(action='ignore', category=FutureWarning)

    logging.basicConfig(level=logging.INFO)

    app()
