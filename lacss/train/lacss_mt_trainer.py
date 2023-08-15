from __future__ import annotations

import jax
import jax.numpy as jnp
import orbax
from tqdm import tqdm

from lacss.deploy import Predictor
from lacss.types import *

from .lacss_trainer import LacssTrainer
from .trainer import _get_iterator


class LacssMTTrainer(LacssTrainer):
    def __init__(
        self,
        config: dict = {},
        collaborator_config: Optional[dict] = None,
        *,
        optimizer: Optional[Optimizer] = None,
        seed: Union[int, Array] = 42,
    ):
        super().__init__(config, collaborator_config, optimizer=optimizer, seed=seed)

    def _merged_ds(self, labeled_dataset, unlabeled_dataset):
        labeled_dataset = _get_iterator(labeled_dataset)
        unlabeled_dataset = _get_iterator(unlabeled_dataset)

        while True:
            yield next(labeled_dataset)

            img = next(unlabeled_dataset)
            if isinstance(img, dict):
                img = img["image"]
            pred = self.predictor.predict(img)
            locs = np.asarray(pred["pred_locations"])
            scores = np.asarray(pred["pred_scores"])
            locs = np.where(scores[:, None] >= 0.5, locs, -1)

            yield dict(
                image=img,
                gt_locations=locs,
            )

    def do_training(
        self,
        datasets: tuple[DataSource, DataSource],
        val_dataset: DataSource = None,
        n_steps: int = 50000,
        validation_interval: int = 5000,
        checkpoint_manager: Optional[orbax.checkpoint.CheckpointManager] = None,
        *,
        warmup_steps: int = 0,
        sigma: float = 20.0,
        pi: float = 2.0,
        alpha: float = 0.999,
    ) -> None:
        pred_model = self.model.principal
        pred_params = self.params["principal"]
        self.predictor = Predictor((pred_model, pred_params))

        combined_dataset = self._merged_ds(*datasets)
        train_iter = self.train(combined_dataset, rng_cols=["droppath"], training=True)

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

            @jax.jit
            def mt_merge(tree_a, tree_b, alpha):
                return jax.tree_util.tree_map(
                    lambda a, b: a * alpha + b * (1 - alpha),
                    tree_a,
                    tree_b,
                )

            print(f"Current step {cur_step} going to {next_cp_step}")
            for _ in tqdm(range(cur_step, next_cp_step)):
                logs = next(train_iter)

                self.predictor.params = mt_merge(
                    self.predictor.params,
                    self.params["principal"],
                    alpha,
                )

            print(", ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))
            self._make_checkpoint(checkpoint_manager)
            self._validate(val_dataset)
