from __future__ import annotations

from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Iterable, Optional, Union

import flax.linen as nn
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm

import lacss.metrics
from lacss.losses import (
    aux_size_loss,
    collaborator_border_loss,
    collaborator_segm_loss,
    lpn_loss,
    segmentation_loss,
)

from ..modules import Lacss, LacssCollaborator
from ..typing import Array, Optimizer
from .base_trainer import Trainer
from .strategy import JIT


class _CKSModel(nn.Module):
    principal: Lacss
    collaborator: LacssCollaborator

    def __call__(self, image, cls_id=None, gt_locations=None, *, training=False):
        outputs = self.principal(
            image,
            gt_locations=gt_locations,
            training=training,
        )

        if self.collaborator is not None:
            outputs.update(self.collaborator(image=image, cls_id=cls_id))
        return outputs


class LacssTrainer:
    """Main trainer class for Lacss"""

    default_lr: float = 0.001

    def __init__(
        self,
        config: dict = {},
        collaborator_config: Optional[dict] = None,
        *,
        optimizer: Optional[Optimizer] = None,
        seed: Union[int, Array] = 42,
        strategy: type = JIT,
    ):
        """Constructor

        Args:
            config: configuration dictionary for Lacss model
            collaborator_config: configuration dictionary for the collaborator model used in
                weakly-supervised training. If set to None, then no collaborator model will be
                created. In this case, training with weak-supervision will result in a error.

        Keyword Args:
            seed: RNG seed
            optimizer: Override the default optimizer
            strategy: Training backend. See See [Traing backends](/api/train/#training-backends).
        """
        principal = Lacss.from_config(config)

        if collaborator_config is None:
            collaborator = None
        else:
            collaborator = LacssCollaborator(**collaborator_config)

        self.model = _CKSModel(
            principal=principal,
            collaborator=collaborator,
        )

        self.optimizer = optimizer or optax.adamw(LacssTrainer.default_lr)
        self.seed = seed
        self.strategy = strategy

    def _train_to_next_interval(self, n_steps, trainer, train_iter, cpm, val_dataset):
        for _ in tqdm(range(n_steps)):
            next(train_iter)

        print("Losses: " + repr(train_iter.loss_logs))

        if cpm is not None:
            cpm.save(
                cpm.latest_step() + 1,
                args=ocp.args.StandardSave(train_iter),
            )

        if val_dataset is not None:
            val_metrics = [
                lacss.metrics.LoiAP([5, 2, 1]),
                lacss.metrics.BoxAP([0.5, 0.75]),
            ]

            var_logs = trainer.compute_metrics(
                val_dataset,
                val_metrics,
                dict(params=train_iter.parameters),
            )

            for k, v in var_logs.items():
                print(f"{k}: {v}")

        train_iter.reset_loss_logs()

    def _get_warmup_trainer(self, sigma, pi):
        pre_seg_loss = partial(segmentation_loss, pretraining=True)
        pre_col_seg_loss = partial(collaborator_segm_loss, sigma=100.0, pi=1.0)
        losses = [
            lpn_loss,
            pre_seg_loss,
            pre_col_seg_loss,
            collaborator_border_loss,
            aux_size_loss,
        ]
        return Trainer(
            model=self.model,
            loss=losses,
            optimizer=self.optimizer,
            seed=self.seed,
            strategy=self.strategy,
        )

    def _get_trainer(self, sigma, pi):
        col_seg_loss = partial(collaborator_segm_loss, sigma=sigma, pi=pi)
        losses = [
            lpn_loss,
            segmentation_loss,
            col_seg_loss,
            collaborator_border_loss,
            aux_size_loss,
        ]
        return Trainer(
            model=self.model,
            loss=losses,
            optimizer=self.optimizer,
            seed=self.seed,
            strategy=self.strategy,
        )

    def do_training(
        self,
        dataset: Iterable,
        val_dataset: Iterable | None = None,
        n_steps: int = 50000,
        validation_interval: int = 5000,
        checkpoint_manager: Optional[ocp.CheckpointManager] = None,
        *,
        warmup_steps: int = 0,
        sigma: float = 20.0,
        pi: float = 2.0,
    ) -> None:
        """Runing training.

        Args:
            dataset: An data iterator feed training data. The data should be in the form of a tuple:
                (x, y).
                    x is a dictionary with at least two keys:
                        "image": Trainging image.
                        "gt_locations": Point labels. Nx2 array
                    y is a dictionary of extra labels. It can be
                        None: for point-supervised training
                        "gt_labels": A index-label image (H, W). For segmentation label.
                        "gt_image_mask": A binary image (H, W). For weakly supervised training
            n_steps: Total training steps
            validation_inteval: Step intervals to perform validation and checkpointing.
            val_dataset: If not None, performing validation on this dataset. The data should be in the
                form of a tuple (x, y):
                    x is a dictionary with one key: "image"
                    y is a dictionary with two lables: "gt_bboxes" and "gt_locations"
            checkpoint_manager: If supplied, will be used to created checkpoints. A checkpoint manager
                can be obtained by calling:
                ```
                options = orbax.CheckpointManagerOptions(...)
                manager = orbax.checkpoint.CheckpointManager(
                    'path/to/directory/',
                    options = options
                )
                ```


        keyword Args:
            warmup_steps: Only used for point-supervised training. Pretraining steps, for which a large
                sigma values is used. This should be multiples of validation_inteval
            sigma: Only for point-supervised training. Expected cell size
            pi: Only for point-supervised training. Amplitude of the prior term.
        """
        train_iter = self.train(dataset, rng_cols=["droppath"], training=True)
        cur_step = 0

        if cur_step < warmup_steps:
            trainer = self._get_warmup_trainer(sigma, pi)
            train_it = trainer.train(dataset, rng_cols=["droppath"], training=True)

            while cur_step < warmup_steps:
                next_cp_step = (
                    (cur_step + validation_interval)
                    // validation_interval
                    * validation_interval
                )

                print(f"Current step {cur_step} going to {next_cp_step}")

                self._train_to_next_interval(
                    next_cp_step - cur_step,
                    trainer,
                    train_it,
                    checkpoint_manager,
                    val_dataset,
                )

                cur_step = next_cp_step

        trainer = self._get_trainer(sigma, pi)
        train_it = trainer.train(
            dataset,
            rng_cols=["droppath"],
            init_vars=dict(params=train_it.parameters),
            training=True,
        )

        while cur_step < n_steps:
            next_cp_step = (
                (cur_step + validation_interval)
                // validation_interval
                * validation_interval
            )

            print(f"Current step {cur_step} going to {next_cp_step}")

            self._train_to_next_interval(
                next_cp_step - cur_step,
                trainer,
                train_it,
                checkpoint_manager,
                val_dataset,
            )

            cur_step = next_cp_step

        self.parameters = train_it.parameters

    def save(self, save_path) -> None:
        """Save a pickled copy of the Lacss model in the form of (module:Lacss, weights:FrozenDict). Only saves the principal model.

        Args:
            save_path: Path to the pkl file
        """
        import pickle

        _module = self.model.principal
        _params = self.parameters["principal"]

        with open(save_path, "wb") as f:
            pickle.dump((_module, _params), f)
