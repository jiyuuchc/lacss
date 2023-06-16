#!/usr/bin/env python

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import dataclasses
import json
import logging
from functools import partial
from os.path import join
from pathlib import Path

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import optax
import typer
from tqdm import tqdm

import lacss
from lacss.train import TFDatasetAdapter

app = typer.Typer(pretty_exceptions_enable=False)

avg_size = {
    "BT474": 24.6,
    "A172": 34.6,
    "MCF7": 18.5,
    "BV2": 13.3,
    "Huh7": 40.8,
    "SkBr3": 21.8,
    "SKOV3": 44.8,
    "SHSY5Y": 20.1,
}


def _get_default_scaling(name, target_size=34.6):
    """Need to be a numpy function due to tf.string limitation"""
    cell_type = name.numpy().split(b"_")[0].decode()
    cellsize = avg_size[cell_type]
    return target_size / cellsize


def train_parser(inputs, target_size=[544, 544]):
    default_scale = tf.py_function(
        _get_default_scaling, [inputs["filename"]], tf.float32
    )

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

    return dict(
        image=tf.ensure_shape(inputs["image"], target_size + [1]),
        gt_locations=inputs["centroids"],
    ), dict(
        gt_bboxes=inputs["bboxes"],
        gt_masks=inputs["masks"],
    )


def train_data(
    datapath: Path,
    n_buckets: int,
    batchsize: int,
):
    ds = lacss.data.dataset_from_coco_annotations(
        datapath / "annotations" / "LIVECell" / "livecell_coco_train.json",
        datapath / "images" / "livecell_train_val_images",
        [520, 704, 1],
    )
    ds = ds.cache(str(datapath / "train_cache")).repeat()
    ds = ds.map(train_parser)

    ds = ds.bucket_by_sequence_length(
        element_length_func=lambda x, _: tf.shape(x["gt_locations"])[0],
        bucket_boundaries=list(np.arange(1, n_buckets + 1) * (4096 // n_buckets) + 1),
        bucket_batch_sizes=(max(batchsize, 1),) * (n_buckets + 1),
        padding_values=-1.0,
        pad_to_bucket_boundary=True,
    )

    if batchsize <= 0:
        ds = ds.unbatch()

    return TFDatasetAdapter(ds, steps=-1).get_dataset()


def val_parser(inputs):
    del inputs["masks"]

    inputs["image"] = tf.image.per_image_standardization(inputs["image"])

    default_scale = tf.py_function(
        _get_default_scaling, [inputs["filename"]], tf.float32
    )

    h, w, _ = inputs["image"].shape
    h = tf.round(h * default_scale)
    w = tf.round(w * default_scale)
    inputs = lacss.data.resize(inputs, target_size=[h, w])

    return dict(
        image=inputs["image"],
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
    config: Path,
    datapath: Path = Path("../../livecell_dataset"),
    logpath: Path = Path("."),
    seed: int = 42,
    batchsize: int = 1,
    n_epochs: int = 10,
    steps_per_epoch: int = 3500 * 5,
    init_epoch: int = 0,
    lr: float = 0.002,
    n_buckets: int = 4,
):
    tf.random.set_seed(seed)

    logging.info(f"Logging to {logpath}")

    train_gen = train_data(
        datapath,
        n_buckets,
        batchsize,
    )
    val_gen = val_data(datapath)

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
