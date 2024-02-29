#!/usr/bin/env python

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import dataclasses
import json
import logging
from pathlib import Path
from pprint import pprint
from typing import Iterator

import optax
import orbax.checkpoint
import typer
from data import augment, get_cell_type_and_scaling, remove_redundant
from flax.core.frozen_dict import freeze, unfreeze

from lacss.data import dataset_from_coco_annotations, resize
from lacss.train import LacssTrainer, TFDatasetAdapter
from lacss.typing import *
from lacss.utils import load_from_pretrained

app = typer.Typer(pretty_exceptions_enable=False)


def train_data(
    datapath: Path,
    *,
    target_size=[544, 544],
    v1_scaling=False,
) -> Iterator:
    def _train_parser(inputs):
        ver = 1 if v1_scaling else 2
        cell_type, default_scale = tf.py_function(
            lambda x: get_cell_type_and_scaling(x, version=ver),
            [inputs["filename"]],
            (tf.int32, tf.float32),
        )

        del inputs["masks"]
        del inputs["bboxes"]

        inputs = augment(inputs, default_scale, target_size)

        n_locs = tf.shape(inputs["centroids"])[0]
        pad_size = (n_locs - 1) // 1024 * 1024 + 1024 - n_locs
        gt_locations = tf.pad(
            inputs["centroids"], [[0, pad_size], [0, 0]], constant_values=-1
        )

        return dict(
            image=tf.ensure_shape(inputs["image"], target_size + [1]),
            gt_locations=gt_locations,
            cls_id=cell_type,
        )

    ds = (
        dataset_from_coco_annotations(
            datapath / "annotations" / "LIVECell" / "livecell_coco_train.json",
            datapath / "images" / "livecell_train_val_images",
            [520, 704, 1],
        )
        .map(remove_redundant)
        .cache(str(datapath / "train_cache"))
        .repeat()
        .map(_train_parser)
        .filter(lambda x: tf.shape(x["gt_locations"])[0] > 0)
        .prefetch(10)
    )

    print("Train dataset is: ")
    pprint(ds.element_spec)

    return TFDatasetAdapter(ds)


def val_data(datapath: Path, *, v1_scaling: bool = False) -> Iterator:
    def _val_parser(inputs):
        del inputs["masks"]

        ver = 1 if v1_scaling else 2
        cell_type, default_scale = tf.py_function(
            lambda x: get_cell_type_and_scaling(x, version=ver),
            [inputs["filename"]],
            (tf.int32, tf.float32),
        )

        inputs["image"] = tf.image.per_image_standardization(inputs["image"])

        h, w, _ = inputs["image"].shape
        h = tf.round(h * default_scale)
        w = tf.round(w * default_scale)
        inputs = resize(inputs, target_size=[h, w])

        return dict(image=inputs["image"], cls_id=cell_type,), dict(
            gt_locations=inputs["centroids"],
            gt_bboxes=inputs["bboxes"],
        )

    ds = (
        dataset_from_coco_annotations(
            datapath / "annotations" / "LIVECell" / "livecell_coco_val.json",
            datapath / "images" / "livecell_train_val_images",
            [520, 704, 1],
        )
        .map(remove_redundant)
        .cache(str(datapath / "val_cache"))
        .map(_val_parser)
        .prefetch(10)
    )

    print("Val dataset is:")
    pprint(ds.element_spec)

    return TFDatasetAdapter(ds)


@app.command()
def run_training(
    config: Path,
    transfer: Path = None,
    datapath: Path = Path("../../livecell_dataset/"),
    logpath: Path = Path("."),
    seed: int = 42,
    n_steps: int = 90000,
    validation_interval: int = 4500,
    warmup_steps: int = 9000,
    lr: float = 0.001,
    offset_sigma: float = 20.0,
    offset_scale: float = 2.0,
    size_loss: float = 0.01,
    v1_scaling: bool = False,
):
    tf.random.set_seed(seed)

    logpath.mkdir(parents=True, exist_ok=True)

    logging.info(f"Logging to {logpath}")

    train_gen = train_data(datapath, v1_scaling=v1_scaling)
    val_gen = val_data(datapath, v1_scaling=v1_scaling)

    with open(config) as f:
        model_cfg = json.load(f)

    logging.info(f"Model configuration loaded from {config}")

    trainer = LacssTrainer(
        model_cfg,
        dict(n_cls=8),
        seed=seed,
    )

    trainer.initialize(train_gen, optax.adamw(lr))

    cp_mngr = orbax.checkpoint.CheckpointManager(
        logpath,
    )

    if transfer is not None:
        _, transfer_params = load_from_pretrained(transfer)
        params = unfreeze(trainer.parameters)
        params["principal"] = transfer_params
        trainer.parameters = freeze(params)

        logging.info(f"Transfer model configuration and weights from {transfer}")

    print("Model configuration:")
    pprint(
        dataclasses.asdict(trainer.model),
        sort_dicts=False,
    )

    trainer.do_training(
        train_gen,
        val_gen,
        n_steps=n_steps,
        warmup_steps=warmup_steps,
        validation_interval=validation_interval,
        checkpoint_manager=cp_mngr,
        sigma=offset_sigma,
        pi=offset_scale,
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    app()
