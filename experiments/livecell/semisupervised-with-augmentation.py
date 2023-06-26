#!/usr/bin/env python

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import dataclasses
import json
import logging
import typing as tp
from functools import partial
from os.path import join
from pathlib import Path

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import optax
import typer
from tqdm import tqdm
from flax.core.frozen_dict import freeze, unfreeze

import lacss
from lacss.train import TFDatasetAdapter
from lacss.deploy import load_from_pretrained
from data import get_cell_type_and_scaling

app = typer.Typer(pretty_exceptions_enable=False)


def train_parser(inputs, target_size=[544, 544]):
    cell_type, default_scale = tf.py_function(
        get_cell_type_and_scaling,
        [inputs["filename"]],
        (tf.int32, tf.float32),
    )

    del inputs["masks"]
    del inputs["bboxes"]

    image = inputs["image"]
    gamma = tf.random.uniform([], 0.5, 2.0)
    image = tf.image.adjust_gamma(image, gamma)
    inputs["image"] = tf.image.per_image_standardization(image)

    inputs = lacss.data.flip_up_down(inputs, p=0.5)
    inputs = lacss.data.flip_left_right(inputs, p=0.5)

    inputs = lacss.data.random_resize(
        inputs, scaling=[default_scale * 0.8, default_scale * 1.2]
    )
    inputs = lacss.data.random_crop_or_pad(
        inputs, target_size=target_size, area_ratio_threshold=0.5
    )

    n_locs = tf.shape(inputs["centroids"])[0]
    pad_size = (n_locs - 1) // 1024 * 1024 + 1024 - n_locs
    gt_locations = tf.pad(
        inputs["centroids"], [[0, pad_size], [0, 0]], constant_values=-1
    )

    return dict(
        image=tf.ensure_shape(inputs["image"], target_size + [1]),
        gt_locations=gt_locations,
        cell_type=cell_type,
    )


def train_data(
    datapath: Path,
):
    ds = lacss.data.dataset_from_coco_annotations(
        datapath / "annotations" / "LIVECell" / "livecell_coco_train.json",
        datapath / "images" / "livecell_train_val_images",
        [520, 704, 1],
    )
    ds = ds.cache(str(datapath / "train_cache")).repeat()
    ds = ds.map(train_parser).filter(lambda x: tf.shape(x["gt_locations"])[0] > 0)

    return TFDatasetAdapter(ds, steps=-1).get_dataset()


def val_parser(inputs):
    del inputs["masks"]

    cell_type, default_scale = tf.py_function(
        get_cell_type_and_scaling,
        [inputs["filename"]],
        (tf.int32, tf.float32),
    )

    inputs["image"] = tf.image.per_image_standardization(inputs["image"])

    h, w, _ = inputs["image"].shape
    h = tf.round(h * default_scale)
    w = tf.round(w * default_scale)
    inputs = lacss.data.resize(inputs, target_size=[h, w])

    return dict(
        image=inputs["image"],
        cell_type=cell_type,
    ), dict(
        gt_locations=inputs["centroids"],
        gt_bboxes=inputs["bboxes"],
    )


def val_data(
    datapath: Path,
):
    ds = lacss.data.dataset_from_coco_annotations(
        datapath / "annotations" / "LIVECell" / "livecell_coco_val.json",
        datapath / "images" / "livecell_train_val_images",
        [520, 704, 1],
    )
    ds = ds.cache(str(datapath / "val_cache"))
    ds = ds.map(val_parser)

    return TFDatasetAdapter(ds, steps=-1).get_dataset()


class CKS(nn.Module):
    lacss: nn.Module
    aux_1: nn.Module
    aux_2: nn.Module

    def __call__(self, image, cell_type, gt_locations=None, *, training=None):
        outputs = self.lacss(image=image, gt_locations=gt_locations, training=training)
        outputs.update(self.aux_1(image, category=cell_type))
        outputs.update(self.aux_2(image, category=cell_type))

        return outputs


@app.command()
def run_training(
    transfer: Path,
    datapath: Path = Path("../../livecell_dataset/"),
    logpath: Path = Path("."),
    seed: int = 42,
    n_epochs: int = 15,
    steps_per_epoch: int = 3500,
    lr: float = 0.001,
    init_epoch: int = 0,
    size_loss: float = 0.01,
    offset_sigma: float = 15.0,
    offset_scale: float = 2.0,
    hard_label: bool = False,
):
    tf.random.set_seed(seed)
    logging.info(f"Logging to {logpath}")

    train_gen = train_data(datapath)
    val_gen = val_data(datapath)

    try:
        with open(transfer) as f:
            model_cfg = json.load(f)
            model = lacss.modules.Lacss.from_config(model_cfg)
            update_params = None

    except:
        model, update_params = load_from_pretrained(transfer)
        if hasattr(model, "lacss"):
            model = model.lacss

    cks_model = CKS(
        model,
        lacss.modules.AuxForeground(
            conv_spec=(16, 32, 64),  # change
            n_groups=8,
        ),
        lacss.modules.AuxInstanceEdge(
            conv_spec=(32, 32),  # change
            n_groups=8,
        ),
    )
    trainer = lacss.train.Trainer(
        model=cks_model,
        losses=[
            lacss.losses.LPNLoss(),
            lacss.losses.SelfSupervisedInstanceLoss(not hard_label),
            lacss.losses.AuxSizeLoss(size_loss),
            lacss.losses.AuxEdgeLoss(),
            lacss.losses.AuxSegLoss(
                offset_sigma=offset_sigma,
                offset_scale=offset_scale,
            ),
        ],
        seed=seed,
        strategy=lacss.train.JIT,
    )

    if not trainer.initialized:
        optimizer = optax.adamw(lr)
        trainer.initialize(train_gen, optimizer)

    print(trainer.model)

    # transfer weights
    if update_params is not None:
        model_params = unfreeze(trainer.params)
        if "lacss" in update_params:
            model_params.update(update_params)
        else:
            model_params.update(
                dict(
                    lacss=update_params,
                )
            )
        trainer.params = freeze(model_params)

    # start training
    logpath.mkdir(parents=True, exist_ok=True)

    train_iter = trainer.train(train_gen, rng_cols=["droppath"], training=True)

    for epoch in range(init_epoch, n_epochs):
        trainer.reset()
        print(f"epoch - {epoch+1}")

        for _ in tqdm(range(steps_per_epoch)):
            logs = next(train_iter)

        print(", ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))

        trainer.checkpoint(logpath / f"cp-{epoch}")

        val_metrics = [
            lacss.metrics.LoiAP([5, 2, 1]),
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
