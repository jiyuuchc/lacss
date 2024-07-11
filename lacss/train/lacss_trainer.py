from __future__ import annotations

from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Iterable, Optional, Union, Sequence

import flax.linen as nn
import jax
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

from ..modules import Lacss, UNet
from ..typing import Array, Optimizer
from xtrain import Trainer, JIT
from ..typing import *

class LacssCollaborator(nn.Module):
    """Collaborator module for semi-supervised Lacss training

    Attributes:
        conv_spec: conv-net specificaiton for cell border predicition
        unet_spec: specification for unet, used to predict cell foreground
        patch_size: patch size for the unet
        n_cls: number of classes (cell types) of input images
    """

    conv_spec: Sequence[int] = (32, 32)
    unet_spec: Sequence[int] = (16, 32, 64)
    patch_size: int = 1
    n_cls: int = 1

    @nn.compact
    def __call__(
        self, image: ArrayLike, cls_id: Optional[ArrayLike] = None
    ) -> DataDict:
        assert cls_id is not None or self.n_cls == 1
        c = cls_id.astype(int).squeeze() if cls_id is not None else 0

        net = UNet(self.unet_spec, self.patch_size)
        unet_out = net(image)

        y = unet_out[0]

        fg = nn.Conv(self.n_cls, (3, 3))(y)
        fg = fg[..., c]

        if fg.shape != image.shape[:-1]:
            fg = jax.image.resize(fg, image.shape[:-1], "linear")

        y = image
        for n_features in self.conv_spec:
            y = nn.Conv(n_features, (3, 3), use_bias=False)(y)
            y = nn.GroupNorm(num_groups=None, group_size=1, use_scale=False)(
                y[None, ...]
            )[0]
            y = jax.nn.relu(y)

        y = nn.Conv(self.n_cls, (3, 3))(y)
        cb = y[..., c]

        return dict(
            fg_pred=fg,
            edge_pred=cb,
        )

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
            outputs["predictions"].update(self.collaborator(image=image, cls_id=cls_id))
        return outputs


class LacssTrainer:
    """Main trainer class for Lacss"""

    default_lr: float = 0.001

    def __init__(
        self,
        model_config: dict,
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
        principal = Lacss.from_config(model_config)

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
                (cpm.latest_step() or 0) + 1,
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
            losses=losses,
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
            losses=losses,
            optimizer=self.optimizer,
            seed=self.seed,
            strategy=self.strategy,
        )

    def get_init_params(self, dataset):
        trainer = self._get_trainer(20.0, 2.0)
        train_it = trainer.train(
            dataset, 
            rng_cols=["droppath"],
            training=True,
        )
        return train_it.parameters


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
        init_vars = None,
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
        cur_step = 0

        if cur_step < warmup_steps:
            trainer = self._get_warmup_trainer(sigma, pi)
            train_it = trainer.train(
                dataset, 
                rng_cols=["droppath"], 
                init_vars=init_vars,
                training=True,
            )

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

                self.parameters = train_it.parameters

            init_vars = dict(params=train_it.parameters)

        trainer = self._get_trainer(sigma, pi)
        train_it = trainer.train(
            dataset,
            init_vars=init_vars,
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
