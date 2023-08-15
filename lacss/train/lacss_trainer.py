from dataclasses import asdict
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax.numpy as jnp
import optax
import orbax.checkpoint
from flax.training import orbax_utils
from tqdm import tqdm

import lacss.losses
import lacss.metrics
from lacss.modules import Lacss, LacssCollaborator
from lacss.types import *

from .strategy import JIT
from .trainer import Trainer


def segmentation_loss(preds, labels, inputs, pretraining=False):
    if labels is None:
        labels = {}

    if "gt_labels" in labels or "gt_masks" in labels:  # supervised
        return lacss.losses.supervised_instance_loss(
            preds=preds,
            labels=labels,
            inputs=inputs,
        )
    elif "gt_image_mask" in labels:  # supervised by point + imagemask
        return lacss.losses.weakly_supervised_instance_loss(
            preds=preds,
            labels=labels,
            inputs=inputs,
        )
    else:  # point-supervised
        return lacss.losses.self_supervised_instance_loss(
            preds=preds, labels=labels, inputs=inputs, soft_label=not pretraining
        )


def collaborator_segm_loss(preds, labels, inputs, sigma, pi):
    if not "fg_pred" in preds:
        return 0.0

    if labels is None:
        labels = {}

    if "gt_image_mask" in labels or "gt_labels" in labels:
        return lacss.losses.supervised_segmentation_loss(
            preds=preds,
            labels=labels,
            inputs=inputs,
        )

    else:
        return lacss.losses.self_supervised_segmentation_loss(
            preds=preds,
            labels=labels,
            inputs=inputs,
            offset_sigma=sigma,
            offset_scale=pi,
        )


def collaborator_border_loss(preds, labels, inputs):
    if not "edge_pred" in preds:
        return 0.0

    if labels is None:
        labels = {}

    if "gt_labels" in labels or "gt_masks" in labels:
        return 0.0

    else:
        return lacss.losses.self_supervised_edge_loss(
            preds=preds,
            labels=labels,
            inputs=inputs,
        )


def mc_loss(preds, labels, inputs):
    if labels is None:
        labels = {}

    if "gt_labels" in labels or "gt_masks" in labels:
        return 0.0
    else:
        return lacss.losses.aux_size_loss(preds=preds, labels=labels, inputs=inputs)


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


class LacssTrainer(Trainer):
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

        super().__init__(
            model=_CKSModel(
                principal=principal,
                collaborator=collaborator,
            ),
            optimizer=optimizer,
            seed=seed,
            strategy=strategy,
        )

        if self.optimizer is None:
            self.optimizer = optax.adamw(LacssTrainer.default_lr)

        # self._cp_step = 0

    def _train_to_next_interval(
        self, train_iter, checkpoint_interval, checkpoint_dir, val_dataset
    ):
        cur_step = self.state.step
        next_cp_step = (
            (cur_step + checkpoint_interval)
            // checkpoint_interval
            * checkpoint_interval
        )

        for _ in tqdm(range(cur_step, next_cp_step)):
            logs = next(train_iter)

        print(", ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))

        if checkpoint_dir is not None:
            cps = [int(p.name.split("-")[-1]) for p in checkpoint_dir.glob("chkpt-*")]
            cp_cnt = max(cps) + 1
            self.checkpoint(checkpoint_dir / f"chkpt-{cp_cnt}")

        if val_dataset is not None:
            val_metrics = [
                lacss.metrics.LoiAP([5, 2, 1]),
                lacss.metrics.BoxAP([0.5, 0.75]),
            ]

            var_logs = self.test_and_compute(val_dataset, metrics=val_metrics)
            for k, v in var_logs.items():
                print(f"{k}: {v}")

    def _validate(self, val_dataset):
        if val_dataset is not None:
            val_metrics = [
                lacss.metrics.LoiAP([5, 2, 1]),
                lacss.metrics.BoxAP([0.5, 0.75]),
            ]

            var_logs = self.test_and_compute(
                val_dataset, metrics=val_metrics, strategy=JIT
            )
            for k, v in var_logs.items():
                print(f"{k}: {v}")

    def _make_checkpoint(self, checkpoint_manager):
        if checkpoint_manager is not None:
            lastest_step = checkpoint_manager.latest_step()
            if lastest_step is None:
                lastest_step = 0

            save_args = orbax_utils.save_args_from_target(self.state)
            checkpoint_manager.save(
                lastest_step + 1,
                {"config": asdict(self.model), "train_state": self.state},
                save_kwargs={"train_state": {"save_args": save_args}},
            )

    def restore_from_checkpoint(
        self, checkpoint_manager: orbax.checkpoint.CheckpointManager, step: int = -1
    ) -> None:
        """Restore train state from a checkpoint.

        Args:
            checkpoint_manager: Orbax checkpoint manager
            step: The exact checkpoint to restore. Latest if unspecified.
        """
        if step < 0:
            step = checkpoint_manager.latest_step()

        # restore_args = orbax_utils.restore_args_from_target(self.state, None)

        restored = checkpoint_manager.restore(
            step,
            items=dict(
                config={},
                train_state=self.state,
            ),
            # restore_kwargs={"train_state": {"restore_args": restore_args}},
        )
        self.state = restored["train_state"]

        # reconstruct model or not?
        # from lacss.utils import dataclass_from_dict
        # self.model = dataclass_from_dict(restored["config"])

        # self._cp_step = step

    def reset(
        self,
        warmup_steps: int = 0,
        sigma: float = 20.0,
        pi: float = 2.0,
    ) -> None:
        cur_step = self.state.step

        if cur_step >= warmup_steps:
            col_seg_loss = partial(collaborator_segm_loss, sigma=sigma, pi=pi)
            col_seg_loss.name = "collaborator_segm_loss"
            self.losses = [
                lacss.losses.lpn_loss,
                segmentation_loss,
                col_seg_loss,
                collaborator_border_loss,
                mc_loss,
            ]
        else:
            pre_seg_loss = partial(segmentation_loss, pretraining=True)
            pre_seg_loss.name = "segmentation_loss"
            pre_col_seg_loss = partial(collaborator_segm_loss, sigma=100.0, pi=1.0)
            pre_col_seg_loss.name = "collaborator_segm_loss"
            self.losses = [
                lacss.losses.lpn_loss,
                pre_seg_loss,
                pre_col_seg_loss,
                collaborator_border_loss,
                mc_loss,
            ]

        super().reset()

    def do_training(
        self,
        dataset: DataSource,
        val_dataset: DataSource = None,
        n_steps: int = 50000,
        validation_interval: int = 5000,
        checkpoint_manager: Optional[orbax.checkpoint.CheckpointManager] = None,
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
                    trainer.get_checkpointer(),
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

        while self.state.step < n_steps:
            cur_step = self.state.step
            next_cp_step = (
                (cur_step + validation_interval)
                // validation_interval
                * validation_interval
            )

            self.reset(
                warmup_steps=warmup_steps,
                sigma=sigma,
                pi=pi,
            )

            print(f"Current step {cur_step} going to {next_cp_step}")
            for _ in tqdm(range(cur_step, next_cp_step)):
                logs = next(train_iter)

            print(", ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))
            self._make_checkpoint(checkpoint_manager)
            self._validate(val_dataset)

    @classmethod
    def get_checkpointer(cls) -> orbax.checkpoint.Checkpointer:
        """Returns a checkpointer obj for this Trainer. Convienent function for creating a checkpoint manager"""

        return {
            "config": orbax.checkpoint.Checkpointer(
                orbax.checkpoint.JsonCheckpointHandler()
            ),
            "train_state": orbax.checkpoint.PyTreeCheckpointer(),
        }

    @classmethod
    def from_checkpoint(cls, cp_path) -> "LacssTrainer":
        """load the module and its weight from a checkpoint
            This utility static method allows use checkpoint as a model save. It ignores
            optstate.

        Args:
            cp_path: checkpoint location (a dir)

        Returns:
            LacssTrainer.
        """
        import json

        cp_path = Path(cp_path)

        with open(cp_path / "config" / "metadata", "r") as f:
            cfg = json.load(f)

        obj = cls(cfg["principal"], cfg["collaborator"])

        fake_data = dict(
            image=jnp.zeros([64, 64, 3]),
            gt_locations=jnp.zeros([8, 2]),
            cls_id=jnp.asarray(0),
        )
        obj.initialize([fake_data], tx=optax.adam(cls.default_lr))

        # restore_args = orbax_utils.restore_args_from_target(obj.state, None)
        obj.state = orbax.checkpoint.PyTreeCheckpointer().restore(
            cp_path / "train_state",
            item=obj.state,
            # restore_args = restore_args,
            # transforms={},
        )

        return obj

    def pickle(self, save_path) -> None:
        """Save a pickled copy of the Lacss model in the form of (model_config:dict, weights:FrozenDict). Only saves the principal model.

        Args:
            save_path: Path to the pkl file
        """
        import pickle

        _cfg = self.model.principal.get_config(0)
        _params = self.params["principal"]

        with open(save_path, "wb") as f:
            pickle.dump((_cfg, _params), f)
