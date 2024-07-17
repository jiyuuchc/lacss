#!/usr/bin/env python

from __future__ import annotations

import logging
from collections.abc import Mapping

from absl import app, flags
from ml_collections import config_flags, ConfigDict

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("initfrom", None, "start training from existing model file")
flags.DEFINE_string("checkpoint", None, "resume from a previous checkpoint")
flags.DEFINE_string("logpath", ".", "logging directory")

def run_training(_):
    from pathlib import Path

    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    import optax
    import pprint
    import tensorflow as tf
    import wandb
    
    from tqdm import tqdm
    from xtrain import Trainer, VMapped, JIT

    from lacss.losses import supervised_instance_loss
    from lacss.metrics import BoxAP
    from lacss.modules import Lacss
    # from livecell_dataset import ds_train, ds_val

    jnp = jax.numpy
    config = _CONFIG.value

    wandb.init(project=config.name, config=config)

    def init_rngs(seed):
        # random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

    print("=========CONFIG==========")
    pprint.pp(config)

    seed = config.train.get("seed", 4242)
    init_rngs(seed)
    logging.info(f"Use RNG seed value {seed}")

    logpath = Path(_FLAGS.logpath)
    logpath.mkdir(parents=True, exist_ok=True)
    logging.info(f"Logging to {logpath.resolve()}")

    wandb.config["logpath"] = str(logpath)

    print("=========DATA============")
    ds_train = config.data.ds_train
    ds_val = config.data.ds_val

    print("Train dataset:")
    pprint.pp(ds_train)
    pprint.pp(ds_val)
        
    print("=========MODEL===========")
    if _FLAGS.initfrom is not None:
        from lacss.utils import load_from_pretrained
        model, params = load_from_pretrained(Path(_FLAGS.initfrom))
        init_vars = dict(params=params)

        wandb.config["init_file"] = _FLAGS.initfrom

    else:
        model_cfg = config.model

        if isinstance(model_cfg, Lacss):
            model = model_cfg
        else:
            model = Lacss.from_config(model_cfg)
            model.integrator.dim_out = config.fpn_dim
            model.detector.conv_spec = (config.fpn_dim,) * 4
            model.segmentor.conv_spec = ((config.fpn_dim,) * 3, (config.fpn_dim//4,))
        init_vars = None

    pprint.pp(model.get_config())

    lr = optax.piecewise_constant_schedule(
        config.train.get("lr", 1e-4),
        {config.train.steps: 0.1}
    )
    optimizer = optax.adamw(lr, config.train.get("weight_decay", 1e-3))

    loss_fns = (
        "losses/lpn_detection_loss",
        "losses/lpn_localization_loss",
        supervised_instance_loss,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        losses=loss_fns,
        seed=seed,
        strategy=VMapped,
    )

    train_it = trainer.train(ds_train, training=True, init_vars=init_vars)

    checkpointer = ocp.StandardCheckpointer()

    if _FLAGS.checkpoint is not None:
        cp_path = Path(_FLAGS.checkpoint).absolute()
        train_it = checkpointer.restore(
            cp_path, args=ocp.args.StandardRestore(train_it)
        )
        logging.info(f"restored checkpoint from {cp_path}")

        wandb.config["checkpoint"] = str(cp_path)

    print("=======TRAINNING=========")
    total_steps = config.train.steps + config.train.get("finetune_steps", config.train.steps//5)
    if total_steps % config.train.validation_interval != 0:
        total_steps = (total_steps // config.train.validation_interval + 1) * config.train.validation_interval

    if not isinstance(ds_val, dict|ConfigDict):
        ds_val = dict(ds_val = ds_val)

    with tqdm(total=total_steps) as pbar:
        while train_it.step < total_steps:
            next(train_it)
            pbar.update(int(train_it.step) - pbar.n)

            if train_it.step % config.train.validation_interval == 0 or train_it.step >= total_steps:
                print(f"Loss at step : {train_it.step} - {train_it.loss_logs}")

                checkpointer.save(
                    logpath.absolute() / f"cp-{train_it.step}",
                    args=ocp.args.StandardSave(train_it),
                )

                metrics = {
                    name: trainer.compute_metrics(
                        ds,
                        BoxAP([0.5, 0.75]),
                        dict(params=train_it.parameters),
                        strategy=JIT,
                    )
                    for name, ds in ds_val.items()
                }

                pprint.pp(metrics)

                wandb.log({
                    "loss": train_it.loss,
                    "metrics": jax.tree.map(lambda v: {"50": v[0], "75": v[1]}, metrics),
                })

                train_it.reset_loss_logs()

    train_it.save_model(logpath/"model_file")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(run_training)
