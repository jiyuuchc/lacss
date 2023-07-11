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
from pprint import pprint

import numpy as np
import optax
import orbax.checkpoint
import typer
from data import augment, get_cell_type_and_scaling, remove_redundant
from flax.core.frozen_dict import freeze, unfreeze
from tqdm import tqdm

import lacss.data
import lacss.train
from lacss.deploy import load_from_pretrained
from lacss.types import *

app = typer.Typer(pretty_exceptions_enable=False)


def train_data(
    datapath: Path,
    n_buckets: int,
    batchsize: int,
    *,
    target_size=[544, 544],
    v1_scaling=False,
):
    v = 1 if v1_scaling else 2

    def _train_parser(inputs):
        _, default_scale = tf.py_function(
            lambda x: get_cell_type_and_scaling(x, v),
            [inputs["filename"]],
            (tf.int32, tf.float32),
        )

        inputs = augment(inputs, default_scale, target_size)

        return dict(
            image=tf.ensure_shape(inputs["image"], target_size + [1]),
            gt_locations=inputs["centroids"],
        ), dict(
            gt_bboxes=inputs["bboxes"],
            gt_masks=inputs["masks"],
        )

    ds = (
        lacss.data.dataset_from_coco_annotations(
            datapath / "annotations" / "LIVECell" / "livecell_coco_train.json",
            datapath / "images" / "livecell_train_val_images",
            [520, 704, 1],
        )
        .map(remove_redundant)
        .cache(str(datapath / "train_cache"))
        .repeat()
        .map(_train_parser)
        .filter(lambda x, _: tf.shape(x["gt_locations"])[0] > 0)
        .bucket_by_sequence_length(
            element_length_func=lambda x, _: tf.shape(x["gt_locations"])[0],
            bucket_boundaries=list(
                np.arange(1, n_buckets + 1) * (4096 // n_buckets) + 1
            ),
            bucket_batch_sizes=(max(batchsize, 1),) * (n_buckets + 1),
            padding_values=-1.0,
            pad_to_bucket_boundary=True,
        )
        .prefetch(10)
    )

    if batchsize <= 0:
        ds = ds.unbatch()

    print("Train dataset is: ")
    pprint(ds.element_spec)

    return lacss.train.TFDatasetAdapter(ds, steps=-1).get_dataset()


def val_data(
    datapath: Path,
    *,
    target_size=[544, 544],
    v1_scaling=False,
):
    v = 1 if v1_scaling else 2

    def _val_parser(inputs, target_size=None):
        _, default_scale = tf.py_function(
            lambda x: get_cell_type_and_scaling(x, v),
            [inputs["filename"]],
            (tf.int32, tf.float32),
        )

        del inputs["masks"]

        inputs["image"] = tf.image.per_image_standardization(inputs["image"])

        h, w, _ = inputs["image"].shape
        h = tf.round(h * default_scale)
        w = tf.round(w * default_scale)
        inputs = lacss.data.resize(inputs, target_size=[h, w])
        if target_size is not None:
            inputs = lacss.data.random_crop_or_pad(inputs, target_size=target_size)

        return dict(image=inputs["image"],), dict(
            gt_locations=inputs["centroids"],
            gt_bboxes=inputs["bboxes"],
        )

    ds = (
        lacss.data.dataset_from_coco_annotations(
            datapath / "annotations" / "LIVECell" / "livecell_coco_val.json",
            datapath / "images" / "livecell_train_val_images",
            [520, 704, 1],
        )
        .map(remove_redundant)
        .cache(str(datapath / "val_cache"))
        .map(_val_parser)
        .filter(lambda x, y: tf.shape(y["gt_locations"])[0] > 0)
        .prefetch(10)
    )

    print("Val dataset is:")
    pprint(ds.element_spec)

    return lacss.train.TFDatasetAdapter(ds, steps=-1).get_dataset()


@app.command()
def run_training(
    config: Path,
    transfer: Path = None,
    datapath: Path = Path("../../livecell_dataset"),
    logpath: Path = Path("."),
    seed: int = 42,
    batchsize: int = 1,
    n_steps: int = 300000,
    validation_interval: int = 15000,
    lr: float = 0.002,
    n_buckets: int = 4,
    v1_scaling: bool = False,
):
    tf.random.set_seed(seed)

    logpath.mkdir(parents=True, exist_ok=True)

    logging.info(f"Logging to {logpath.resolve()}")

    train_gen = train_data(
        datapath,
        n_buckets,
        batchsize,
        v1_scaling=v1_scaling,
    )

    val_gen = val_data(datapath, v1_scaling=v1_scaling)

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
        trainer.get_checkpointer(),
    )

    if len(cp_mngr.all_steps()) > 0:
        trainer.restore_from_checkpoint(cp_mngr)

    elif transfer is not None:
        _, params = load_from_pretrained(transfer)
        orig_params = unfreeze(trainer.params)
        orig_params["principal"] = params
        trainer.params = freeze(orig_params)

        logging.info(f"Transfer model weights from {transfer}")

    print("Model configuration:")
    pprint(
        dataclasses.asdict(trainer.model),
        sort_dicts=False,
    )

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
