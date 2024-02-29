#!/usr/bin/env python
from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import optax
import orbax.checkpoint
import typer
from data import tfds_from_data_path

import lacss.data
import lacss.train
from lacss.utils import load_from_pretrained

app = typer.Typer(pretty_exceptions_enable=False)


def train_parser(inputs):
    image = inputs["image"]
    gt_locations = inputs["centroids"]
    mask_labels = tf.cast(inputs["label"], tf.float32)

    if tf.random.uniform([]) >= 0.5:
        image = tf.image.transpose(image)
        gt_locations = gt_locations[..., ::-1]
        mask_labels = tf.transpose(mask_labels)

    x_data = dict(
        image=image,
        gt_locations=gt_locations,
    )
    y_data = dict(
        gt_labels=mask_labels,
    )

    return x_data, y_data


def train_data(datapath, n_buckets, batchsize, ch=0):
    ds_train = (
        tfds_from_data_path(datapath / "train", ch=ch)
        # .cache(str(datapath / "train_cache"))
        .repeat()
        .map(train_parser)
        .filter(lambda x, _: tf.size(x["gt_locations"]) > 0)
        .bucket_by_sequence_length(
            element_length_func=lambda x, _: tf.shape(x["gt_locations"])[0],
            bucket_boundaries=list(
                np.arange(1, n_buckets + 1) * (2560 // n_buckets) + 1
            ),
            bucket_batch_sizes=(batchsize,) * (n_buckets + 1),
            padding_values=-1.0,
            pad_to_bucket_boundary=True,
        )
    )

    return lacss.train.TFDatasetAdapter(ds_train, steps=-1).get_dataset()


def val_parser(data):
    return (
        dict(
            image=data["image"],
        ),
        dict(
            gt_bboxes=data["bboxes"],
            gt_locations=data["centroids"],
            # gt_labels=data["label"],
        ),
    )


def val_data(datapath, ch=0):
    ds_val = (
        tfds_from_data_path(datapath / "val", imgsize=[256, 256, 2], ch=ch)
        # .cache(str(datapath / "val_cache"))
        .map(val_parser)
    )

    return lacss.train.TFDatasetAdapter(ds_val).get_dataset()


@app.command()
def run_training(
    config: Path,
    transfer: Path = None,
    datapath: Path = Path("../../tissue_net"),
    logpath: Path = Path("."),
    seed: int = 42,
    batchsize: int = 1,
    n_steps: int = 100000,
    validation_interval: int = 10000,
    lr: float = 0.002,
    n_buckets: int = 4,
    nucleus: bool = False,
):
    tf.random.set_seed(seed)

    logpath.mkdir(parents=True, exist_ok=True)

    logging.info(f"Logging to {logpath.resolve()}")

    train_gen = train_data(
        datapath,
        n_buckets,
        batchsize,
        ch=1 if nucleus else 0,
    )

    val_gen = val_data(
        datapath,
        ch=1 if nucleus else 0,
    )

    schedules = [optax.cosine_decay_schedule(lr * batchsize, validation_interval)] * 100
    boundaries = [validation_interval * i for i in range(1, 100)]
    schedule = optax.join_schedules(schedules, boundaries)

    with open(config) as f:
        model_cfg = json.load(f)

    logging.info(f"Model configuration loaded from {config}")

    trainer = lacss.train.LacssTrainer(
        model_cfg,
        seed=seed,
        strategy=lacss.train.VMapped,
    )

    trainer.initialize(train_gen, optax.adamw(schedule))

    cp_mngr = orbax.checkpoint.CheckpointManager(
        logpath,
    )

    if transfer is not None:
        _, params = load_from_pretrained(transfer)
        trainer.parameters = params

        logging.info(f"Transfer model configuration and weights from {transfer}")

    print(dataclasses.asdict(trainer.model))

    trainer.do_training(
        train_gen,
        val_gen,
        n_steps=n_steps,
        validation_interval=validation_interval,
        checkpoint_manager=cp_mngr,
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    app()
