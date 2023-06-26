#!/usr/bin/env python

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import dataclasses
import itertools
import os
import pickle
from functools import partial

from os.path import join

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState
from jax.config import config
from pathlib import Path
from tqdm import tqdm

import lacss

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    from . import data
except:
    import data

import typer

app = typer.Typer(pretty_exceptions_enable=False)

SIGMAS = (15, 15, 10, 15, 15, 12, 15, 12)


class FgAcc:
    def __init__(self):
        self.cnts = 0.0
        self.acc = 0.0

    def update(self, inputs, preds, gt_labels, **kwargs):
        pred_masks = preds["fg_pred"]
        pred_masks = (pred_masks >= 0).astype(int)
        gt_masks = (gt_labels > 0).astype(int)
        acc = (pred_masks == gt_masks).mean()

        self.acc += acc
        self.cnts += 1

    def compute(self):
        return self.acc / self.cnts


edge_net_configs = (
    (32, 32),
    (32, 32, 32),
    (64, 64, 128, 128),
)

net_configs = (
    (16, 32, 64),
    (16, 32, 64, 128),
    (16, 32, 64, 128, 256),
)


@app.command()
def run_training(
    transfer: str = "../../cnsp4.pkl",
    datapath: Path = Path("../../livecell_dataset/"),
    logpath: Path = Path("."),
    seed: int = 42,
    batchsize: int = 1,
    n_epochs: int = 15,
    warmup_epochs: int = 0,
    steps_per_epoch: int = 3500,
    lr: float = 0.001,
    init_epoch: int = 0,
    size_loss: float = 0.01,
    n_buckets: int = 4,
    cell_type: int = -1,
    offset_scale: float = 2.0,
    dp_rate: float = 0.2,
    edge_net_config: int = 0,
    net_config: int = 0,
    share_weights: bool = False,
):
    import lacss.deploy

    tf.random.set_seed(seed)

    steps_per_epoch = steps_per_epoch // batchsize

    try:
        os.makedirs(logpath)
    except:
        pass

    train_data = data.train_data(datapath, n_buckets, batchsize, cell_type=cell_type)
    val_data = data.val_data(datapath, cell_type=cell_type, batch=False)

    # model

    with open(transfer, "rb") as f:
        cp = pickle.load(f)
        if isinstance(cp, lacss.train.Trainer):
            trainer = cp

        else:
            lacss_cfg, params = cp

            if isinstance(lacss_cfg, nn.Module):
                lacss_cfg = dataclasses.asdict(lacss_cfg)
            lacss_cfg = unfreeze(lacss_cfg)

            if "params" in params:
                params = params["params"]
            params = freeze(params)

            lacss_cfg["backbone"]["drop_path_rate"] = dp_rate

            cfg = dict(
                cfg=lacss_cfg,
                aux_edge_cfg=dict(
                    conv_spec=edge_net_configs[edge_net_config],
                    n_groups=8,
                )
                if not share_weights
                else None,
                aux_fg_cfg=dict(
                    n_groups=8,
                    conv_spec=net_configs[net_config],
                    share_weights=share_weights,
                ),
            )
            model = lacss.modules.lacss.LacssWithHelper(**cfg)

            trainer = lacss.train.Trainer(
                model=model,
                optimizer=optax.adamw(lr * batchsize),
                losses=[],
                seed=seed,
                strategy=lacss.train.strategy.VMapped,
            )

            trainer.initialize(val_data)

            new_params = unfreeze(trainer.params)
            new_params.update(dict(_lacss=params))
            trainer.state = trainer.state.replace(params=freeze(new_params))

    file_writer = tf.summary.create_file_writer(str(logpath / "train"))
    to_label = lacss.ops.patches_to_label

    def _epoch_end(epoch, logs=None, eval=False):
        print(f"epoch - {epoch}")
        if logs is not None:
            print(", ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))

        with file_writer.as_default():
            for k, v in logs.items():
                tf.summary.scalar(k, v, epoch)

        # trainer.save_model(logpath/f"weight-{epoch}.pkl", "_lacss")
        trainer.checkpoint(logpath / f"cp-{epoch}")
        trainer.reset()

        if eval:
            val_metrics = [
                lacss.metrics.LoiAP([0.2, 0.5, 1.0]),
                lacss.metrics.BoxAP([0.5, 0.75]),
                FgAcc(),
            ]
            var_logs = trainer.test_and_compute(
                val_data, val_metrics, strategy=lacss.train.JIT
            )
            for k, v in var_logs.items():
                print(f"{k}: {v}")

        # for c in range(8) if cell_type == -1 else [cell_type]:
        # data = itertools.dropwhile(lambda x: x["category"] != c, val_data())
        # inputs, label = next(data)
        # preds = trainer(inputs, strategy=lacss.train.JIT)
        # label = to_label(preds, inputs["image"].shape[:2])
        # output = jnp.concatenate(
        # [
        # inputs["image"][None,...]
        # label[None,...,None] / label.max(),
        # jax.nn.sigmoid(preds["fg_pred"][None, ..., None]),
        # ],
        # axis=2,
        # )

    #
    # with file_writer.as_default():
    # tf.summary.image(f"img_{c}", output, epoch)

    epoch = init_epoch

    # warmup

    if epoch < warmup_epochs:
        trainer.losses = [
            lacss.losses.LPNLoss(),
            lacss.losses.SelfSupervisedInstanceLoss(False),
            lacss.losses.AuxSizeLoss(0.02),
            lacss.losses.AuxEdgeLoss(),
            lacss.losses.AuxSegLoss(
                offset_sigma=100.0,
                offset_scale=1.0,
            ),
        ]

        pb = tqdm(
            trainer.train(train_data, rng_cols=["droppath", "augment"], training=True)
        )

        for steps, logs in enumerate(pb):
            if (steps + 1) % steps_per_epoch == 0:
                epoch += 1
                _epoch_end(epoch, logs)

                if epoch >= warmup_epochs:
                    break

    # train
    if epoch < n_epochs:
        trainer.losses = [
            lacss.losses.LPNLoss(),
            lacss.losses.SelfSupervisedInstanceLoss(),
            lacss.losses.AuxSizeLoss(size_loss),
            lacss.losses.AuxEdgeLoss(),
            lacss.losses.AuxSegLoss(
                offset_sigma=SIGMAS,
                offset_scale=offset_scale,
            ),
        ]

        pb = tqdm(
            trainer.train(train_data, rng_cols=["droppath", "augment"], training=True)
        )
        for steps, logs in enumerate(pb):
            if (steps + 1) % steps_per_epoch == 0:
                epoch += 1
                _epoch_end(epoch, logs, eval=True)

                if epoch >= n_epochs:
                    break


if __name__ == "__main__":
    # config.update("jax_debug_nans", True)
    app()
