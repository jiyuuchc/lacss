import dataclasses
import pathlib
import typing as tp
from functools import lru_cache, partial

import cloudpickle
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState
from optax import GradientTransformation

from ..utils import Inputs, _get_name
from . import strategy
from .data import *
from .loss import Loss, LossLog

# so that multiple calls return the same obj
# this avoids JIT when supplying partial func as args
_cached_partial = lru_cache(partial)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        losses: tp.Optional[tp.Union[tp.Sequence[Loss], Loss]] = None,
        optimizer: GradientTransformation = None,
        seed: int = 42,
        strategy: type = strategy.JIT,
    ):

        self.model = model
        self.losses = losses
        self._loss_weights = None

        self.seed = seed if isinstance(seed, jnp.ndarray) else jax.random.PRNGKey(seed)
        self._optimizer = optimizer

        self._strategy = strategy
        self._initialized = False

        # self.reset()

    def reset(self, loss_weights=None):

        if self.losses is None:
            raise ValueError(f"No loss functions provided")

        losses = self.losses
        try:
            iter(losses)
        except:
            losses = (losses,)

        if loss_weights is not None:
            self._loss_weights = loss_weights
        else:
            loss_weights = self._loss_weights

        if loss_weights is None:
            loss_weights = (1.0,) * len(losses)

        if len(loss_weights) != len(self.losses):
            raise ValueError(
                f"Loss weights supplied {loss_weights} does not match the number of loss functions ({len(losses)})"
            )

        self.loss_logs = tuple(
            LossLog(loss, w) for loss, w in zip(self.losses, loss_weights)
        )

    @property
    def initialized(self):
        return self._initialized

    def initialize(self, dataset, tx: GradientTransformation = None):

        if tx is None:
            tx = self.optimizer

        if not self._initialized:

            if self.model.scope is None:

                peek = next(dataset())
                inputs, _, _ = unpack_x_y_sample_weight(peek)

                self.seed, key = jax.random.split(self.seed)
                variables = self._strategy.init_fn(key, self.model, inputs)

            else:

                self.model, variables = self.model.unbind()

            self.state = TrainState.create(
                apply_fn=self.model.apply,
                params=variables["params"],
                tx=tx,
            )

        else:

            raise ValueError("Calling initialize() on already initialized Trainer")

        self._initialized = True

    def __call__(self, inputs, *, strategy=None, **kwargs):
        if strategy is None:
            strategy = self._strategy

        predict_fn = strategy.predict

        state = self.state
        if len(kwargs) > 0:
            state = state.replace(
                apply_fn=_cached_partial(self.state.apply_fn, **kwargs)
            )
        preds = predict_fn(state, inputs)

        return preds

    def compute_loss_log(self):
        return {
            _get_name(loss_log.loss_fn): loss_log.compute()
            for loss_log in self.loss_logs
        }

    def train(self, dataset, strategy=None, rng_cols=None, **kwargs):
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

            state = self.state.replace(
                apply_fn=_cached_partial(self.state.apply_fn, **kwargs)
            )
            state, self.loss_logs, preds = train_fn(
                state, self.loss_logs, inputs, labels, rngs
            )
            self.state = state.replace(apply_fn=self.state.apply_fn)

            batch_logs = self.compute_loss_log()

            yield batch_logs

    def save_model(self, path, sub_module: tp.Optional[str] = None):
        module = self.model
        params = self.params

        if sub_module is not None:
            module = module.bind(dict(params=params))
            module = getattr(module, sub_module)
            module, params = module.unbind()
            params = params["params"]

        if isinstance(path, str):
            path = pathlib.Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            cfg = dataclasses.asdict(module)
            params = unfreeze(params)
            cloudpickle.dump((cfg, params), f)

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

        trainer = cloudpickle.loads(_bytes)

        if not isinstance(trainer, Trainer):
            raise TypeError("The saved obj is not a Trainer checkpoint")

        return trainer

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

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, tx):

        self._optimizer = tx

        if self._initialized:

            self.state = TrainState.create(
                apply_fn=self.model.apply,
                params=self.params,
                tx=tx,
            )
