import pathlib
import typing as tp
from functools import partial

import cloudpickle
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from optax import GradientTransformation

from ..utils import Inputs, _get_name
from . import strategy
from .data import *
from .loss import Loss
from .pytree import Pytree, static_field
from .wrapper import *


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        losses: tp.Optional[tp.Union[tp.Sequence[Loss], Loss]] = None,
        optimizer: GradientTransformation = None,
        seed: int = 42,
        train_strategy: type = strategy.JIT,
    ):
        self.model = model
        self.loss_log = LossLog(losses)
        self.seed = seed if isinstance(seed, jnp.ndarray) else jax.random.PRNGKey(seed)
        self.optimizer = optimizer

        self._strategy = train_strategy
        self._initialized = False

    def reset(self):
        self.loss_log.reset()

    @property
    def initialized(self):
        return self._initialized

    def initialize(self, dataset, tx: GradientTransformation = None):

        if tx is None:
            tx = self.optimizer

        if not self._initialized:
            peek = next(dataset())
            inputs, _, _ = unpack_x_y_sample_weight(peek)

            self.seed, key = jax.random.split(self.seed)
            self.state = self._strategy.init_fn(key, self.model, inputs, tx)

        self.reset()
        self._initialized = True

    def __call__(self, inputs, *, strategy=None):
        if strategy is None:
            strategy = self._strategy

        predict_fn = strategy.predict
        preds = predict_fn(self.state, inputs)
        return preds

    def train(self, dataset, strategy=None, rng_cols=None):
        if strategy is None:
            strategy = self._strategy

        if not self._initialized:
            raise ValueError("Try to run uninitialized trainer")

        self.reset()

        self.seed, seed = jax.random.split(self.seed)
        for step, data in enumerate(dataset()):
            inputs, labels, _ = unpack_x_y_sample_weight(data)
            if rng_cols is not None:
                key = jax.random.fold_in(seed, step)
                keys = jax.random.split(key, len(rng_cols))
                rngs = {name: k for name, k in zip(rng_cols, keys)}
            else:
                rngs = None
            train_fn = strategy.train_step
            self.state, self.loss_log, preds = train_fn(
                self.state, self.loss_log, inputs, labels, rngs
            )

            batch_logs = self.loss_log.compute()

            yield batch_logs

    def checkpoint(self, path):
        if isinstance(path, str):
            path = pathlib.Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def from_checkpoint(path):
        if isinstance(path, str):
            path = pathlib.Path(path)

        try:
            _bytes = path.read_bytes()
        except BaseException as e:
            raise OSError(f"Could not load the checkpoint. Got exception: {e}")

        return cloudpickle.loads(_bytes)

    def test(self, dataset, metrics, strategy=None):
        if strategy is None:
            strategy = self._strategy

        try:
            iter(metrics)
        except TypeError:
            metrics = [metrics]

        # self.initialize(dataset, strategy)

        for data in dataset():
            inputs, labels, _ = unpack_x_y_sample_weight(data)
            predict_fn = strategy.predict
            preds = predict_fn(self.state, inputs)
            kwargs = dict(
                inputs=inputs,
                preds=preds,
                **labels,
            )
            for m in metrics:
                m.update(**kwargs)
            yield metrics

    def test_and_compute(self, *args, **kwargs):
        for metrics in self.test(*args, **kwargs):
            pass
        return {_get_name(m): m.compute() for m in metrics}

    @property
    def params(self):
        return self.state.params
