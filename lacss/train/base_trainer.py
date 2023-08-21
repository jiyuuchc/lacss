from __future__ import annotations

import pathlib
import pickle
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState

from ..typing import *
from .loss import LossLog
from .strategy import JIT
from .utils import Inputs, _get_name

LOSSES = Union[LossFunc, Sequence[LossFunc]]
METRICS = Union[Metric, Sequence[Metric]]

# so that multiple calls return the same obj
# this avoids JIT when supplying partial func as args
_cached_partial = lru_cache(partial)


class Trainer:
    """A general purpose FLAX model trainer. Help avoiding most of the biolerplate code when trainning with FLAX.

    Attributes:
        model: A Flax module
        losses: A list of loss function (or other callabels).
        optimizer: An optax optimizer
        seed: RNG seed
        params: Current model parameters. This is a frozen dict. This is read/write.
        initialized: Whether the model has been initialized with an optimizer and initial weights.

    Example:

        ```
        trainer = lacss.train.Trainer(my_module, my_loss_func)

        trainer.initialize(my_dataset, tx=my_optimzier)

        train_it = trainer.train(my_dataset)

        for k in range(train_steps):
            loss_logs = next(train_it)
            if k % 1000 == 0:
                print(loss_logs)
                trainer.reset()
        ```

    """

    def __init__(
        self,
        model: nn.Module,
        losses: Optional[LOSSES] = None,
        optimizer: Optional[Optimizer] = None,
        seed: Union[int, Array] = 42,
        strategy: type = JIT,
    ):
        """Constructor

        Args:
            model: A Flax module. It can be a bound module with parameters.
            losses: A loss function (callabels) or a list of loss functions, or None.
                These should have the call signiture of:
                    loss = loss_fn(inputs, labels, preds)
                The "preds" is the model output. The "inputs" and "labels" are from the train dataset.
                The model trains on the sum of all losses.
            optimizer: An optax optimzier
            seed: RNG seed.
            strategy: Training backend. See [Traing backends](./#training-backends).

        """
        self.model = model
        self.losses = losses

        self.seed = seed if isinstance(seed, jnp.ndarray) else jax.random.PRNGKey(seed)
        self._optimizer = optimizer

        self._strategy = strategy
        self._initialized = False

    def reset(
        self,
        loss_weights: Optional[Sequence[float]] = None,
        losses: Optional[LOSSES] = None,
    ):
        """Reset internal loss value tracking

        Args:
            loss_weights: Optional weights of individual loss functions. If not None, the
                total loss is weighted sum.
            losses: Optionally override the class losses defined as the object initialization.
        """
        if losses is None:
            losses = self.losses

        if losses is None:
            raise ValueError(f"No loss functions provided")

        try:
            iter(losses)
        except:
            losses = (losses,)

        if loss_weights is None:
            loss_weights = (1.0,) * len(losses)

        if len(loss_weights) != len(losses):
            raise ValueError(
                f"Loss weights supplied {loss_weights} does not match the number of loss functions ({len(losses)})"
            )

        self.loss_logs = tuple(
            LossLog(loss, w) for loss, w in zip(losses, loss_weights)
        )

    @property
    def initialized(self):
        return self._initialized

    def initialize(self, data: Iterable, tx: Optional[Optimizer] = None) -> None:
        """Initialize the model weights and optimizer states.

        Args:
            data: An iterator or iterable to produce training dataset. It is not
                used if model is bound with weights already.
                see [train()](./#lacss.train.base_trainer.Trainer.train)
            tx: Optional optax optimzier for when the object was constructed without an
                optimizer
        """
        if tx is None:
            tx = self.optimizer

        if not self._initialized:
            if self.model.scope is None:
                peek = next(iter(data))
                # inputs, _, _ = unpack_x_y_sample_weight(peek)
                inputs = peek[0] if isinstance(peek, tuple) else peek

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

    def __call__(
        self, inputs: Any, *, strategy: Optional[type] = None, **kwargs
    ) -> Any:
        """Calling the underlying model

        Args:
            inputs: A tuple or a dict as the inputs for the model.
            strategy: Optionally override the default strategy.
            **kwargs: Additional keyword args passed on to the model

        Returns:
            Model outputs.
        """
        if strategy is None:
            strategy = self._strategy

        predict_fn = strategy.predict

        state = self.state.replace(apply_fn=_cached_partial(self.model.apply, **kwargs))
        preds = predict_fn(state, inputs)

        return preds

    def compute_loss_log(self) -> dict:
        return {
            _get_name(loss_log.loss_fn): loss_log.compute()
            for loss_log in self.loss_logs
        }

    def train(
        self,
        dataset: Iterable,
        strategy: Optional[type] = None,
        rng_cols: Sequence[str] = None,
        **kwargs,
    ) -> Iterator:
        """Create the training iterator

        Args:
            dataset: An iterator or iterable to supply the training data.
                The dataset should produce ```(inputs, labels, sample_weight)```, however
                both the labels and the sample_weight are optional. The inputs is either a tuple
                or a dict. If the inputs is a dict, the keys are interpreted as the names for
                keyword args of the model's __call__ function.
            strategy: Optionally override the default strategy.
            rng_cols: Names of any RNG used by the model. Should be a list of strings.
            **kwargs: Additional keyward args passed to the model. E.g. "training=True"

        Returns:
            A iterator. Stepping through the iterator will train the model. The iterator
                itself returns a loss log dict, which are mean loss values for each loss
                function.

        """
        if strategy is None:
            strategy = self._strategy

        if not self._initialized:
            raise ValueError("Try to run uninitialized trainer")

        train_fn = strategy.train_step

        self.state = self.state.replace(
            apply_fn=_cached_partial(self.model.apply, **kwargs)
        )

        self.reset()

        self.seed, seed = jax.random.split(self.seed)
        for step, data in enumerate(dataset):
            # batch = unpack_x_y_sample_weight(data)
            batch = data

            if rng_cols is not None:
                key = jax.random.fold_in(seed, step)
                keys = jax.random.split(key, len(rng_cols))
                rngs = {name: k for name, k in zip(rng_cols, keys)}
            else:
                rngs = None

            self.state, self.loss_logs, preds = train_fn(
                self.state, self.loss_logs, batch, rngs
            )

            batch_logs = self.compute_loss_log()

            yield batch_logs

    def save_model(self, path: PathLike, sub_module: Optional[str] = None) -> None:
        """Save the model in a pickled file. The pickle is a tuple of
            (module, weights).

        Args:
            path: The file path.
            sub_module: Optionally only save a sub_module of the model
                by specifying the name
        """
        module = self.model
        params = self.params

        if sub_module is not None:
            module = module.bind(dict(params=params))
            module = getattr(module, sub_module)
            module, params = module.unbind()
            params = params["params"]

        if isinstance(path, str):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump((module, params), f)

    def pickle_self(self, path: PathLike) -> None:
        """Make a pickle save of the trainer. This saves the model as well as
        the training states.

        Args:
            path: The file path.
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def restore_from_pickle(path: PathLike):
        """Restore from the a pickled saved.

        Args:
            path: The pickle file path.

        Returns:
            A new trainer object.
        """
        path = pathlib.Path(path)

        try:
            _bytes = path.read_bytes()
        except BaseException as e:
            raise OSError(f"Could not load the checkpoint. Got exception: {e}")

        trainer = pickle.loads(_bytes)

        if not isinstance(trainer, Trainer):
            raise TypeError("The saved obj is not a Trainer checkpoint")

        return trainer

    def test(
        self,
        dataset: Iterable,
        metrics: METRICS,
        strategy: Optional[type] = None,
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
            strategy: Optionally override the default strategy.

        Returns:
            An iterator. Stepping through it will drive the updating of each metric
                obj. The iterator itself return the list of metrics.
        """
        if strategy is None:
            strategy = self._strategy

        try:
            iter(metrics)
        except TypeError:
            metrics = [metrics]

        state = self.state.replace(apply_fn=_cached_partial(self.model.apply, **kwargs))
        predict_fn = strategy.predict

        for data in dataset:
            # inputs, labels, _ = unpack_x_y_sample_weight(data)
            inputs = data[0] if isinstance(data, tuple) else data
            preds = predict_fn(state, inputs)
            kwargs = dict(
                batch=data,
                prediction=preds,
            )
            for m in metrics:
                m.update(**kwargs)

            yield metrics

    def test_and_compute(self, *args, **kwargs) -> dict:
        """A convient function to compute all metrics. See [test() fucntion](./#lacss.train.base_trainer.Trainer.test)

        Returns:
            A metric dict. Keys are metric names.
        """
        for metrics in self.test(*args, **kwargs):
            pass
        return {_get_name(m): m.compute() for m in metrics}

    @property
    def params(self) -> Params:
        return self.state.params

    @params.setter
    def params(self, new_params: Params) -> None:
        old_state = self.state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=new_params,
            tx=old_state.tx,
        )

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, tx: Optimizer) -> None:
        self._optimizer = tx

        if self._initialized:
            self.state = TrainState.create(
                apply_fn=self.model.apply,
                params=self.params,
                tx=tx,
            )
