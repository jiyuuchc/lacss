import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import dataclasses
import logging
import pickle
from functools import partial
from logging.config import valid_ident
from os.path import join

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from flax.core.frozen_dict import freeze, unfreeze
from jax.config import config
from tqdm import tqdm

import lacss
from lacss.utils import show_images

app = typer.Typer(pretty_exceptions_enable=False)

import data


@app.command()
def run_training(
    datapath: str = "../../../tissue_net/",
    transfer: str = "../livecell/runs/supervised/convnext_p2/convnext_p2.pkl",
    logpath: str = ".",
    seed: int = 123,
    batchsize: int = 1,
    n_epochs: int = 15,
    warmup_epochs: int = 1,
    steps_per_epoch: int = 3000,
    lr: float = 0.001,
    init_epoch: int = 0,
    size_loss: float = 0.005,
    n_buckets: int = 4,
    offset_sigma: float = 5.0,
    offset_scale: float = 2.0,
    use_attention: bool = False,
    dp_rate: float = 0.2,
):
    # data
    train_data = data.train_data(datapath, n_buckets, batchsize)
    val_data = data.val_data(datapath, n_buckets, 4)

    # model
    with open(transfer, "rb") as f:
        module, params = pickle.load(f)
    module_cfg = dataclasses.asdict(module)
    module_cfg["backbone_cfg"]["drop_path_rate"] = dp_rate

    cfg = dict(
        cfg=module_cfg,
        aux_edge_cfg=dict(n_groups=6),
        aux_fg_cfg=dict(
            n_groups=6,
            start_layer=2,
            conv_spec=(16, 32, 64, 128),
            use_attention=use_attention,
            augment=True,
        ),
    )
    model = lacss.modules.lacss.LacssWithHelper(**cfg)

    logging.info(dataclasses.asdict(model))

    losses = [
        lacss.losses.LPNLoss(),
        lacss.losses.AuxEdgeLoss(),
        lacss.losses.InstanceOverlapLoss(),
        lacss.losses.AuxSizeLoss(size_loss),
        lacss.losses.AuxSegLoss(
            ver=1, offset_scale=offset_scale, offset_sigma=offset_sigma
        ),
    ]

    steps_per_epoch = steps_per_epoch // batchsize

    trainer = lacss.train.Trainer(
        model=model,
        optimizer=optax.adamw(lr * batchsize),
        losses=losses,
        seed=seed,
        strategy=lacss.train.strategy.VMapped,
    )

    trainer.initialize(val_data)

    new_params = unfreeze(trainer.params)
    new_params.update(dict(_lacss=params))
    trainer.state = trainer.state.replace(params=freeze(new_params))

    # warmup

    file_writer = tf.summary.create_file_writer(join(logpath, "train"))
    samples = data.Samples(datapath)
    to_label = jax.vmap(partial(lacss.ops.patches_to_label, input_size=(255, 255)))

    def _epoch_end(epoch):
        print(f"epoch - {epoch}")
        print(", ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))

        trainer.checkpoint(join(logpath, f"cp-{epoch}"))
        trainer.reset()

        val_metrics = [
            lacss.metrics.LoiAP([0.2, 0.5, 1.0]),
            lacss.metrics.BoxAP([0.5, 0.75]),
            # lacss.metrics.MaskAP(
            #     [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
            # ),
        ]
        var_logs = trainer.test_and_compute(val_data, val_metrics)
        for k, v in var_logs.items():
            print(f"{k}: {v}")

        preds = samples.sample_predict(trainer)
        label = to_label(preds)
        with file_writer.as_default():
            label = label[..., None]
            label = label / label.max(axis=(1, 2), keepdims=True)
            tf.summary.image("Label", label, step=epoch, max_outputs=8)
            tf.summary.image(
                "FG",
                jax.nn.sigmoid(preds["fg_pred"][..., None]),
                step=epoch,
                max_outputs=8,
            )

    epoch = init_epoch
    pb = tqdm(
        trainer.train(train_data, rng_cols=["droppath", "augment"], training=True)
    )
    for steps, logs in enumerate(pb):
        if epoch >= warmup_epochs:
            break

        if (steps + 1) % steps_per_epoch == 0:
            epoch += 1
            _epoch_end(epoch)

    # train
    trainer.losses = [
        lacss.losses.LPNLoss(),
        lacss.losses.AuxEdgeLoss(),
        lacss.losses.AuxSegLoss(
            ver=1, offset_scale=offset_scale, offset_sigma=offset_sigma
        ),
        lacss.losses.SelfSupervisedInstanceLoss(ver=2),
        lacss.losses.AuxSizeLoss(size_loss),
    ]
    trainer.reset()

    pb = tqdm(
        trainer.train(train_data, rng_cols=["droppath", "augment"], training=True)
    )
    for steps, logs in enumerate(pb):
        if epoch >= n_epochs:
            break

        if (steps + 1) % steps_per_epoch == 0:
            epoch += 1
            _epoch_end(epoch)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    app()
