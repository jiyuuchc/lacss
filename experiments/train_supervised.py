#!/usr/bin/env python

from __future__ import annotations

import logging
import random
import pickle
from collections.abc import Mapping
from pathlib import Path
from functools import partial

from absl import app, flags
from ml_collections import config_flags, ConfigDict

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("initfrom", None, "start training from existing model file")
flags.DEFINE_string("checkpoint", None, "resume from a previous checkpoint")
flags.DEFINE_string("logpath", "", "logging directory")


def _instance_loss(batch, prediction):
    from lacss.losses import supervised_instance_loss, mean_over_boolean_mask
    from lacss.ops import iou_loss
    import jax
    import jax.numpy as jnp
    from xtrain import unpack_x_y_sample_weight

    _, label, _ = unpack_x_y_sample_weight(batch)

    if "gt_masks" in label:
        return supervised_instance_loss(batch, prediction)

    preds = prediction["predictions"]
    bbox_regrs = preds["bbox_regressions"]
    instance_mask = preds["segmentation_is_valid"]
    n_patches = instance_mask.shape[0]

    ps_z, _, ps = preds["segmentations"].shape[-3:]
    assert ps_z == 1 # only valid for 2d for now

    yx = jnp.c_[preds["segmentation_y0_coord"], preds["segmentation_x0_coord"]] + ps / 2
    yx = yx + jax.nn.tanh(bbox_regrs[:, 0, :2]) * ps
    sz = jnp.exp(bbox_regrs[:, 0, 2:]) * 35
    pred_bboxes = jnp.c_[yx - sz/2, yx + sz/2]

    loss = iou_loss(label['gt_bboxes'][:n_patches], pred_bboxes)

    return mean_over_boolean_mask(loss, instance_mask)

def run_training(_):
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    import optax
    import pprint
    import tensorflow as tf
    import wandb
    
    from tqdm import tqdm
    from xtrain import Trainer, VMapped, JIT

    from lacss.metrics import BoxAP, LoiAP
    from lacss.modules import Lacss
    from lacss.losses import supervised_instance_loss
    from lacss.train.train import train_fn
    from lacss.utils import load_from_pretrained

    jnp = jax.numpy
    config = _CONFIG.value

    wandb.init(project=config.name, config=config)

    def init_rngs(seed):
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

    print("=========CONFIG==========")
    pprint.pp(config)

    seed = config.train.get("seed", 4242)
    init_rngs(seed)
    logging.info(f"Use RNG seed value {seed}")

    if _FLAGS.checkpoint is not None and _FLAGS.logpath=="":
        logpath = Path(_FLAGS.checkpoint).parent
    else:
        if _FLAGS.logpath == "":
            run_name = wandb.run.name
            if run_name is not None and run_name != "":
                run_name = run_name.split("-")
                run_name = "-".join(run_name[-1:]+run_name[:-1])
                logpath = Path(config.name) / run_name
            else:
                from datetime import datetime
                logpath = Path(config.name) / datetime.now().strftime("%y%m%d%H%M")
        else:
            logpath = Path(_FLAGS.logpath)
        logpath.mkdir(parents=True, exist_ok=True)

    logging.info(f"Logging to {logpath.resolve()}")
    wandb.config["logpath"] = str(logpath)

    print("=========DATA============")
    ds_train = (
        config.data.ds_train
        .batch(config.data.batch_size)
        .prefetch(1)
    )
    ds_val = config.data.ds_val

    print("Train dataset:")
    pprint.pp(ds_train)
    pprint.pp(ds_val)

    print("=========MODEL===========")
    if _FLAGS.initfrom is not None:
        model, params = load_from_pretrained(Path(_FLAGS.initfrom))
        init_vars = dict(params=params)

        wandb.config["init_file"] = _FLAGS.initfrom

    elif _FLAGS.checkpoint is not None:
        _path = Path(_FLAGS.checkpoint).parent
        with open(_path / "model.pkl", "rb") as f:
            model = pickle.load(f)
        init_vars = None

    else:
        model_cfg = config.model

        if isinstance(model_cfg, Lacss):
            model = model_cfg
        else:
            model = Lacss.from_config(model_cfg)

        init_vars = None

    if "backbone_dropout" in config.train:
        model.backbone.drop_path_rate = config.train.backbone_dropout

    with open(logpath / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    pprint.pp(model.get_config())

    print("=========TRAINER===========")

    lr = optax.piecewise_constant_schedule(
        config.train.get("lr", 1e-4),
        {config.train.steps: 0.1}
    )
    optimizer = optax.adamw(lr, config.train.get("weight_decay", 1e-3))

    loss_fns = (
        "losses/lpn_detection_loss",
        "losses/lpn_localization_loss",
        supervised_instance_loss,
        # _instance_loss,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        losses=loss_fns,
        seed=seed,
        strategy=VMapped,
    )

    method = partial(train_fn, config=config.train.config)
    train_it = trainer.train(ds_train, init_vars=init_vars, method=method)

    for froze_part in config.train.get("freeze", []):
        logging.info(f"freeze component: {froze_part}")
        train_it.freeze(froze_part)

    checkpointer = ocp.StandardCheckpointer()

    if _FLAGS.checkpoint is not None:
        cp_path = Path(_FLAGS.checkpoint).absolute()
        train_it = checkpointer.restore(
            cp_path, args=ocp.args.StandardRestore(train_it)
        )
        logging.info(f"restored checkpoint from {cp_path}")

        wandb.config["checkpoint"] = str(cp_path)

    elif "param_override" in config.train:
        _, params = load_from_pretrained(config.train.param_override)
        for k in params:
            logging.info(f"override parameters of submodule: {k}")
            train_it.parameters[k] = params[k]

    print("=======TRAINNING=========")
    total_steps = config.train.steps + config.train.get("finetune_steps", 0)

    if not isinstance(ds_val, dict|ConfigDict):
        ds_val = dict(ds_val = ds_val)

    with tqdm(total=total_steps) as pbar:
        train_it.reset_loss_logs()

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
                        ds, [BoxAP([0.5, 0.75]), LoiAP([5])],
                        dict(params=train_it.parameters),
                        strategy=JIT,
                    )
                    for name, ds in ds_val.items()
                }

                pprint.pp(metrics)

                wandb.log({
                    "loss": train_it.loss,
                    "metrics": metrics,
                })

                train_it.reset_loss_logs()

    train_it.save_model(logpath/"final_model_save.pkl")

if __name__ == "__main__":
    import os
    import jax

    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
    )
    jax.config.update("jax_compilation_cache_dir", "jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)

    logging.basicConfig(level=logging.INFO)
    app.run(run_training)
