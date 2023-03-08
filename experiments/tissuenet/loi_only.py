import warnings
from logging.config import valid_ident

warnings.simplefilter(action="ignore", category=FutureWarning)

import io
import json
import os
import pickle
from functools import partial
from os.path import join

import jax
import numpy as np
import optax
import treex as tx
from tqdm import tqdm

jnp = jax.numpy

import typer

import lacss

app = typer.Typer(pretty_exceptions_enable=False)

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from data import *
from util import *


class ForegroundPredict(tx.Module):
    @tx.compact
    def __call__(self, image):
        orig = image
        x = tx.Conv(24, (3, 3), strides=(2, 2), use_bias=False)(image)
        x = tx.LayerNorm(use_scale=False)(x)
        x = jax.nn.relu(x)
        for ch in (64, 64):
            x = tx.Conv(ch, (3, 3), use_bias=False)(x)
            x = tx.LayerNorm(use_scale=False)(x)
            x = jax.nn.relu(x)
        x = lacss.modules.se_net.SpatialAttention()(x)
        x = jax.image.resize(x, orig.shape[:-1] + (1,), "cubic")
        x = jnp.concatenate([orig, x], axis=-1)
        x = tx.Conv(6, (3, 3))(x)

        return x


class SegLoss(tx.Loss):
    def call(self, preds, tissue_type):
        @jax.vmap
        def _to_patch(img, yc, xc):
            img = jnp.pad(img, [[48, 48], [48, 48]], constant_values=-1.0)
            return img[yc + 48, xc + 48]

        fg = jax.vmap(lambda y, k: y[..., k])(preds["fg_pred"], tissue_type)
        fg = jax.lax.stop_gradient(fg)
        weights = jax.nn.tanh(fg)
        yc = preds["instance_yc"]
        xc = preds["instance_xc"]
        weight_patch = _to_patch(weights, yc, xc)
        loss = -preds["instance_output"] * weight_patch
        loss = loss.sum(where=preds["instance_mask"])
        loss += jnp.count_nonzero(fg > 0)
        loss = loss / (jnp.count_nonzero(preds["instance_mask"]) + 1e-8)
        loss = loss / 96 / 96

        return loss


class AuxSegLoss(tx.Loss):
    def call(self, preds, tissue_type):
        @jax.vmap
        def _to_patch(img, yc, xc):
            img = jnp.pad(img, [[48, 48], [48, 48]], constant_values=-1.0)
            return img[yc + 48, xc + 48]

        @jax.vmap
        def _max_merge(pred):
            label = jnp.zeros([512, 512])
            label = jnp.pad(label, [[48, 48], [48, 48]])
            label -= 1e8
            yc, xc = pred["instance_yc"], pred["instance_xc"]
            label = label.at[yc + 48, xc + 48].max(pred["instance_logit"])
            label = label[48:-48, 48:-48]
            return label

        fg = jax.vmap(lambda y, k: y[..., k])(preds["fg_pred"], tissue_type)
        fg = jax.nn.tanh(fg)
        fg2 = _max_merge(preds)
        loss = 1.0 - jax.nn.tanh(fg2) * fg
        loss = loss.mean(axis=(1, 2))

        return loss


def _to_mask(img, pred):
    label = jnp.zeros(img.shape[:-1])
    label = jnp.pad(label, [[48, 48], [48, 48]])
    yc, xc = pred["instance_yc"], pred["instance_xc"]
    label = label.at[yc + 48, xc + 48].max(pred["instance_output"])
    label = label[48:-48, 48:-48]
    return label


class LOIMaskPredLoss(tx.Loss):
    def call(self, inputs, preds, tissue_type):

        mask_from_instances = jax.vmap(_to_mask)(inputs["image"], preds) >= 0.5
        mask_from_instances = jax.lax.stop_gradient(mask_from_instances)

        fg = jax.vmap(lambda y, k: y[..., k])(preds["fg_pred"], tissue_type)

        p_t = mask_from_instances * (-fg) + (1 - mask_from_instances) * (fg)
        p_t = jnp.exp(p_t) + 1.0
        bce = jnp.log(p_t).mean(axis=(1, 2))

        return bce


class LOILacss(tx.Module):
    def __init__(self, lacss_module, fg_module, **kwargs):
        super().__init__(**kwargs)
        self.lacss_module = lacss_module
        self.fg_module = fg_module

        assert lacss_module.initialized
        assert fg_module.initialized

        self._initialized = True

    def __call__(self, image, gt_locations=None):
        preds = self.lacss_module(image, gt_locations)
        preds.update(dict(fg_pred=self.fg_module(image)))
        return preds


@app.command()
def run_training(
    checkpoint: str,
    datapath: str = "../tissue_net/",
    logpath: str = "./",
    n_epochs: int = 30,
    init_epoch: int = 0,
    size_loss: float = 1e-3,
    lr: float = 1e-3,
    seed: int = 42,
):
    with open(checkpoint, "rb") as f:
        cp = pickle.load(f)
    if isinstance(cp, tx.Module):
        lacss_module = cp
        lacss_module.detector._config_dict["test_max_output"] = 1024
        fg_module = ForegroundPredict().init(
            key=1323, inputs=(jnp.zeros([1, 512, 512, 2]))
        )
        loi_model = LOILacss(lacss_module, fg_module)

        trainer = lacss.train.Trainer(
            model=loi_model,
            losses=[
                lacss.losses.LPNLoss(),
                SizeLoss(size_loss),
                lacss.losses.InstanceOverlapLoss(),
                lacss.losses.InstanceEdgeLoss(),
                AuxSegLoss(),
            ],
            optimizer=optax.adamw(lr),
            seed=seed,
        )
    else:
        trainer = cp

    assert isinstance(trainer, lacss.train.Trainer)

    ds_train = partial(train_dataset, data_path=datapath, supervised=False)
    ds_val = partial(val_dataset, data_path=datapath)
    test_metrics = tx.metrics.Metrics(
        [lacss.metrics.LoiAP([0.2, 0.5, 1.0]), lacss.metrics.BoxAP([0.5, 0.75])]
    )

    file_writer = tf.summary.create_file_writer(join(logpath, "train"))
    with file_writer.as_default():
        tf.summary.image("inputs", sample_images(datapath), step=0, max_outputs=8)

    trainer.model.unfreeze(inplace=True)
    for epoch in range(init_epoch + 1, n_epochs + 1):
        print(f"epoch-{epoch}")
        for logs in tqdm(trainer(ds_train)):
            pass
        print(", ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))

        trainer.checkpoint(join(logpath, f"cp-{epoch}"))

        val_logs = trainer.test_and_compute(ds_val, test_metrics, lacss.train.Core)
        for k, v in val_logs.items():
            print(f"{k}: {v}")

        preds = sample_prediction(datapath, trainer.model.train())
        label = to_label(preds)[..., None]
        label = label / label.max()
        with file_writer.as_default():
            tf.summary.image("fg_pred", preds["fg_pred"], step=epoch, max_outputs=8)
            tf.summary.image("label_pred", label, step=epoch, max_outputs=8)


if __name__ == "__main__":
    app()
