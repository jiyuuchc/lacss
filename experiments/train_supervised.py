#!/usr/bin/env python

from __future__ import annotations

import logging

from absl import app, flags
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", None, "resume from a previous checkpoint")
flags.DEFINE_string("logpath", ".", "logging directory")


def run_training(_):
    import pprint
    from pathlib import Path

    import jax
    import orbax.checkpoint as ocp
    import tensorflow as tf
    
    from xtrain import TFDatasetAdapter, Trainer, VMapped, JIT
    from lacss.losses import supervised_instance_loss
    from lacss.metrics import BoxAP

    config = _CONFIG.value
    pprint.pp(config)

    seed = config.train.seed
    logpath = Path(_FLAGS.logpath)

    logpath.mkdir(parents=True, exist_ok=True)

    logging.info(f"Logging to {logpath.resolve()}")

    tf.random.set_seed(seed)

    ds_train = TFDatasetAdapter(
        config.dataset.train_dataset.repeat().batch(config.train.batchsize).prefetch(1)
    )

    ds_test = TFDatasetAdapter(config.dataset.val_dataset.prefetch(1))

    loss_fns = config.train.get(
        "losses",
        (
            "losses/lpn_localization_loss",
            "losses/lpn_detection_loss",
            jax.vmap(supervised_instance_loss),
        ),
    )

    trainer = Trainer(
        model=config.model,
        optimizer=config.train.optimizer,
        losses=loss_fns,
        seed=seed,
        strategy=VMapped,
    )

    train_it = trainer.train(ds_train, rng_cols=["dropout"], training=True)

    checkpointer = ocp.StandardCheckpointer()

    if _FLAGS.checkpoint is not None:
        train_it = checkpointer.restore(
            Path(_FLAGS.checkpoint).absolute(), args=ocp.args.StandardRestore(train_it)
        )

    while train_it.step < config.train.n_steps:
        next(train_it)

        step = train_it.step

        if step % config.train.validation_interval == 0:
            print(f"Loss at step : {step} - {train_it.loss_logs}")

            checkpointer.save(
                logpath.absolute() / f"cp-{step}",
                args=ocp.args.StandardSave(train_it),
            )

            train_it.reset_loss_logs()

            print(
                trainer.compute_metrics(
                    ds_test,
                    BoxAP([0.5, 0.75]),
                    dict(params=train_it.parameters),
                    strategy=JIT,
                )
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(run_training)
