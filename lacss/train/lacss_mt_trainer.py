from __future__ import annotations

from typing import Iterable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import orbax
from tqdm import tqdm

from ..deploy import Predictor
from ..typing import Array, Optimizer
from ..utils import pack_x_y_sample_weight, unpack_x_y_sample_weight
from .lacss_trainer import LacssTrainer


def _get_iterator(g):
    try:
        it = iter(g)
    except:
        it = iter(g())

    return it


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

            x, y, sw = unpack_x_y_sample_weight(next(unlabeled_dataset))

            if isinstance(x, dict):
                x = x["image"]

            pred = self.predictor.predict(x)
            locs = np.asarray(pred["pred_locations"])
            scores = np.asarray(pred["pred_scores"])
            locs = np.where(scores[:, None] >= 0.5, locs, -1)

            x = dict(
                image=x,
                gt_locations=locs,
            )
            yield pack_x_y_sample_weight(x, y, sw)

    def do_training(
        self,
        datasets: tuple[Iterable, Iterable],
        val_dataset: Iterable = None,
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
