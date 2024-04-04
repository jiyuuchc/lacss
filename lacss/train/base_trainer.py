from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Iterator
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Iterable, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.core.scope import CollectionFilter
from flax.training.train_state import TrainState

from ..typing import *
from .loss import LossLog
from .strategy import JIT
from .utils import (
    Peekable,
    _get_name,
    unpack_prediction_and_state,
    unpack_x_y_sample_weight,
)

WeightedLossFunc = LossFunc | tuple[LossFunc, float | np.number]
LOSSES = Union[LossFunc, Sequence[LossFunc]]
METRICS = Union[Metric, Sequence[Metric]]
RNG = Array

# so that multiple calls return the same obj
# this avoids JIT when supplying partial func as args
_cached_partial = lru_cache(partial)


@struct.dataclass
class TrainIterator(Iterator):
    """The iterator obj returned by Trainer.train(). Iterating this object drives the training. The object supports orbax checkpointing.

    Example:
        ```
        import orbax.checkpoint as ocp

        train_it = trainer.train(dataset)

        # make a checkpoint
        checkpointer = ocp.StandardCheckpointer()
        cp_path = cp_path.absolute() # orbax needs absolute path
        checkpointer.save(cp_path / "test_cp", train_it)

        # restore
        restored = checkpointer.restore(
            cp_path / "test_cp",
            train_it,
        )
        ```

    A caveat is that the checkpoint does not save the current state of the dataset.

    """

    ctx: Trainer = struct.field(pytree_node=False)
    data: Iterator = struct.field(pytree_node=False)
    train_state: TrainState
    rngs: dict[str, RNG]
    loss_logs: tuple[LossLog]
    variables: dict = struct.field(default_factory=dict)

    @property
    def parameters(self):
        return self.train_state.params

    @property
    def step(self):
        return self.train_state.step

    @property
    def loss(self):
        return self._compute_loss_log()

    @property
    def has_aux(self):
        return self.ctx.mutable or self.ctx.capture_intermediates

    def _compute_loss_log(self) -> dict:
        return {
            _get_name(loss_log.loss_fn): loss_log.compute()
            for loss_log in self.loss_logs
        }

    def reset_loss_logs(self):
        """Reset internal loss value tracking.

        Args:
            loss_weights: Optional weights of individual loss functions. If not None, the
                total loss is weighted sum.
        """
        for loss in self.loss_logs:
            loss.reset()

    def __next__(self):
        train_fn = self.ctx.strategy.train_step

        batch = next(self.data)

        train_state, loss_logs, preds = train_fn(self, batch)

        preds, variables = unpack_prediction_and_state(preds, self.has_aux)

        # use the object hack until we upgrade flax
        object.__setattr__(self, "train_state", train_state)
        object.__setattr__(self, "loss_logs", loss_logs)
        object.__setattr__(self, "variables", variables)

        # batch_logs = self._compute_loss_log()

        # return batch_logs

        return preds

    def save_model(self, path: PathLike, sub_module: str | None = None) -> None:
        """Save the model in a pickled file. The pickle is a tuple of
            (module, weights).

        Args:
            path: The file path.
            sub_module: Optionally only save a sub_module of the model
                by specifying the name
        """
        module = self.ctx.model
        params = self.parameters

        if sub_module is not None:
            module = module.bind(dict(params=params))
            module = getattr(module, sub_module)
            module, variables = module.unbind()
            params = variables["params"]

        if isinstance(path, str):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump((module, params), f)


@dataclasses.dataclass
class Trainer:
    """A general purpose FLAX model trainer. Help avoiding most of the biolerplate code when trainning with FLAX.

    Attributes:
        model: A Flax module
        losses: A collection of loss function ( loss_fn(batch:Any, prediction:Any)->float ).
        optimizer: An optax optimizer
        seed: RNG seed
        strategy: a training strategy type

    Example:

        ```
        trainer = lacss.train.Trainer(my_module, my_loss_func)

        train_it = trainer.train(my_dataset)

        for k in range(train_steps):
            _ = next(train_it)
            if k % 1000 == 0:
                print(train_it.loss_logs)
                train_it.reset_loss_logs()
        ```

    """

    model: nn.Module
    losses: LOSSES
    optimizer: Optimizer
    mutable: CollectionFilter = False
    capture_intermediates: Union[bool, Callable[["Module", str], bool]] = False
    seed: int | RNG = 42
    strategy: type = JIT

    def _initialize(self, rng: RNG, data: Iterator) -> dict:
        peek = data.peek()
        inputs, _, _ = unpack_x_y_sample_weight(peek)

        return self.strategy.init_fn(rng, self.model, inputs)

    def train(
        self,
        dataset: Iterable,
        strategy: type | None = None,
        rng_cols: Sequence[str] = [],
        init_vars: dict | None = None,
        frozen: dict | None = None,
        **kwargs,
    ) -> TrainIterator:
        """Create the training iterator

        Args:
            dataset: An iterator or iterable to supply the training data.
                The dataset should produce ```(inputs, labels, sample_weight)```, however
                both the labels and the sample_weight are optional. The inputs is either a tuple
                or a dict. If the inputs is a dict, the keys are interpreted as the names for
                keyword args of the model's __call__ function.
            strategy: Optionally override the default strategy.
            rng_cols: Names of any RNG used by the model. Should be a list of strings.
            init_vars: optional variables to initialize model
            frozen: a pytree indicating frozen parameters
            **kwargs: Additional keyward args passed to the model. E.g. "training=True"

        Returns:
            TrainIterator. Stepping through the iterator will train the model.

        """
        config = dataclasses.replace(self, strategy=strategy or self.strategy)
        assert config.strategy is not None

        dataset_iter = Peekable(iter(dataset))

        seed = (
            self.seed
            if isinstance(self.seed, jnp.ndarray)
            else jax.random.PRNGKey(self.seed)
        )

        rng_cols = rng_cols or []

        seed, key = jax.random.split(seed)
        keys = jax.random.split(key, len(rng_cols))
        rngs = dict(zip(rng_cols, keys))

        if init_vars is None:
            seed, key = jax.random.split(seed)
            keys = jax.random.split(seed, len(rng_cols) + 1)
            init_rngs = dict(zip(rng_cols + ["params"], keys))

            init_vars = self._initialize(init_rngs, dataset_iter)

        if frozen is None:
            tx = self.optimizer
        else:
            frozen = jax.tree_util.tree_map(lambda _: False, init_vars["params"])
            optimizers = {True: optax.set_to_zero(), False: self.optimizer}
            tx = optax.multi_transform(
                optimizers,
                frozen,
            )

        params = init_vars.pop("params")
        train_state = TrainState.create(
            apply_fn=_cached_partial(
                self.model.apply,
                mutable=self.mutable,
                capture_intermediates=self.capture_intermediates,
                **kwargs,
            ),
            params=params,
            tx=tx,
        )

        losses = self.losses
        try:
            iter(losses)
        except:
            losses = (losses,)
        loss_logs = tuple(LossLog(loss) for loss in losses)

        return TrainIterator(
            ctx=config,
            data=dataset_iter,
            train_state=train_state,
            rngs=rngs,
            loss_logs=loss_logs,
            variables=init_vars,
        )

    def test(
        self,
        dataset: Iterable,
        metrics: METRICS,
        variables: dict,
        strategy: type | None = None,
        **kwargs,
    ) -> Iterator:
        """Create test/validation iterator.

        Args:
            dataset: An iterator or iterable to supply the testing data.
                The iterator should yield a tupple of (inputs, labels).
            metrics: A list of Metric objects. They should have two functions:
                m.update(preds, **kwargs):
                    preds is the model output. the remaining kwargs are content of
                    labels.
                m.compute():
                    which should return the accumulated metric value.
            variables: Model weights etc. typically get from TrainIterator
            strategy: Optionally override the default strategy.

        Returns:
            An iterator. Stepping through it will drive the updating of each metric
                obj. The iterator itself return the list of metrics.
        """
        if strategy is None:
            strategy = self.strategy

        try:
            iter(metrics)
        except TypeError:
            metrics = [metrics]

        apply_fn = _cached_partial(self.model.apply, **kwargs)

        predict_fn = strategy.predict

        for data in dataset:
            inputs, _, _ = unpack_x_y_sample_weight(data)
            preds = predict_fn(apply_fn, variables, inputs)
            kwargs = dict(
                batch=data,
                prediction=preds,
            )
            for m in metrics:
                m.update(**kwargs)

            yield metrics

    def compute_metrics(self, *args, **kwargs) -> dict:
        """A convient function to compute all metrics. See [test() fucntion](./#lacss.train.base_trainer.Trainer.test)

        Returns:
            A metric dict. Keys are metric names.
        """
        for metrics in self.test(*args, **kwargs):
            pass
        return {_get_name(m): m.compute() for m in metrics}
