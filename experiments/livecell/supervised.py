#!/usr/bin/env python

import dataclasses
import json
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

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)


try:
    from . import data
except:
    import data

import typer

app = typer.Typer(pretty_exceptions_enable=False)

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


def train_parser_supervised(inputs):
    inputs = lacss.data.parse_train_data_func_full_annotation(
        inputs, target_height=544, target_width=544
    )

    image = inputs["image"]
    gt_locations = inputs["locations"]
    mask_labels = tf.cast(inputs["mask_labels"], tf.float32)

    if tf.random.uniform([]) >= 0.5:
        image = tf.image.transpose(image)
        gt_locations = gt_locations[..., ::-1]
        mask_labels = tf.transpose(mask_labels)

    x_data = dict(
        image=image,
        gt_locations=gt_locations,
    )
    y_data = dict(
        mask_labels=mask_labels,
    )

    return x_data, y_data


def val_parser(inputs):
    return (
        dict(
            image=inputs["image"],
        ),
        dict(
            gt_boxes=inputs["bboxes"],
            gt_locations=inputs["locations"],
        ),
    )


def prepare_data(datapath, celltype, batchsize, n_buckets=16):
    ds_train = data.livecell_dataset_from_tfrecord(join(datapath, "train.tfrecord"))

    if celltype >= 0:
        ds_train = ds_train.filter(lambda x: x["cell_type"] == celltype)

    ds_train = ds_train.map(train_parser_supervised)
    ds_train = ds_train.filter(lambda x, _: tf.size(x["gt_locations"]) > 0).repeat()
    ds_train = ds_train.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x["gt_locations"])[0],
        bucket_boundaries=list(np.arange(1, n_buckets) * (4096 // n_buckets) + 1),
        bucket_batch_sizes=(batchsize,) * n_buckets,
        padding_values=-1.0,
        pad_to_bucket_boundary=True,
    )

    ds_val = data.livecell_dataset_from_tfrecord(join(datapath, "val.tfrecord"))
    ds_val = ds_val.map(val_parser).batch(1)

    return ds_train, ds_val


def get_model(cmd, config, batchsize, seed):
    n_epochs = 10
    steps_per_epoch = 3500 * 5
    schedules = [
        optax.cosine_decay_schedule(0.002 * batchsize, steps_per_epoch)
    ] * n_epochs
    boundaries = [steps_per_epoch * i for i in range(1, n_epochs)]
    lr = optax.join_schedules(schedules, boundaries)

    if cmd == "resume":
        cp = lacss.trainer.Trainer.from_checkpoint(config)
        assert isinstance(cp, lacss.trainer.Trainer)
        trainer = cp
        print(f"Loaded checkpoint {config}")
        try:
            init_epoch = int(config.split("-")[-1])
        except:
            init_epoch = 0
    else:
        if cmd == "transfer":
            import pickle

            model = pickle.load(open(config, "rb"))
            assert isinstance(model, lacss.modules.Lacss)
        elif cmd == "config":
            with open(config) as f:
                model_cfg = json.load(f)
            model = lacss.modules.Lacss(**model_cfg)
        else:
            raise ValueError('Cmd musst be one of "resume", "transfer" or "config"')

        loss = [
            lacss.losses.LPNLoss(),
            lacss.losses.SupervisedInstanceLoss(),
        ]

        trainer = lacss.train.Trainer(
            model=model,
            losses=loss,
            optimizer=optax.adamw(lr),
            seed=seed,
            strategy=lacss.train.strategy.VMapped,
        )

        init_epoch = 0

    return trainer, init_epoch, n_epochs, steps_per_epoch


@app.command()
def run_training(
    cmd: str,
    config: str,
    datapath: str = "../../livecell_dataset",
    logpath: str = "",
    celltype: int = -1,
    seed: int = 42,
    batchsize: int = 1,
):
    tf.random.set_seed(seed)

    if logpath == "":
        import time

        logpath = join(".", time.strftime("%Y%m%d-%H%M%S"))
    try:
        os.makedirs(logpath)
    except:
        pass
    print(f"Logging to {logpath}")

    ds_train, ds_val = prepare_data(datapath, celltype, batchsize)
    print(ds_train.element_spec)

    trainer, init_epoch, n_epochs, steps_per_epoch = get_model(
        cmd, config, batchsize, seed
    )
    print(json.dumps(dataclasses.asdict(trainer.model), indent=2, sort_keys=True))

    epoch = init_epoch
    train_gen = lacss.train.TFDatasetAdapter(ds_train, steps=-1).get_dataset()
    val_gen = lacss.train.TFDatasetAdapter(ds_val).get_dataset()

    if not trainer.initialized:
        trainer.initialize(train_gen)

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
                lacss.metrics.LoiAP([0.1, 0.2, 0.5, 1.0]),
                lacss.metrics.BoxAP([0.5, 0.75]),
            ]
            var_logs = trainer.test_and_compute(val_gen, metrics=val_metrics)

            print(", ".join([f"{k}:{v}" for k, v in var_logs.items()]))

            if epoch == n_epochs:
                break


if __name__ == "__main__":
    app()
