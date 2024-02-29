from __future__ import annotations

import pickle
import warnings
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState

from lacss.utils import unpack_x_y_sample_weight

from ..typing import *
from .loss import LossLog
from .strategy import JIT
from .utils import Inputs, _get_name

WeightedLossFunc = LossFunc | tuple[LossFunc, float | np.number]
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
        losses: LOSSES | None = None,
        optimizer: Optional[Optimizer] = None,
        seed: Union[int, Array] = 42,
        strategy: type = JIT,
        *,
        parameters: dict | None = None,
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
        if model.scope is not None:
            model, variables = model.unbind()
            if parameters is None and "params" in variables:
                parameters = variables["params"]

        self.model = model
        self.losses = losses

        self.seed = seed if isinstance(seed, jnp.ndarray) else jax.random.PRNGKey(seed)
        self.optimizer = optimizer

        self._strategy = strategy

        self.parameters = parameters

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
        return self.parameters is not None

    def initialize(self, data: Iterable, tx: Optional[Optimizer] = None) -> None:
        """Generate initial model weights. Does nothing if self.parameters is not None.

        Args:
            data: An iterable to produce training dataset.
                see [train()](./#lacss.train.base_trainer.Trainer.train)
            tx: Depreciated. Optional optax optimzier for when the object was constructed without an
                optimizer
        """
        if tx is not None:
            self.optimizer = tx
            warnings.warn(
                f"Using tx parameter in initialize() call is deprecated",
                DeprecationWarning,
            )

        if self.parameters is None:
            peek = next(iter(data))
            inputs, _, _ = unpack_x_y_sample_weight(peek)
            # inputs = peek[0] if isinstance(peek, tuple) else peek

            self.seed, key = jax.random.split(self.seed)
            variables = self._strategy.init_fn(key, self.model, inputs)
            self.parameters = variables["params"]

        # self.state = TrainState.create(
        #     apply_fn=self.model.apply,
        #     params=variables["params"],
        #     tx=tx,
        # )

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

        apply_fn = _cached_partial(self.model.apply, **kwargs)
        preds = predict_fn(apply_fn, self.parameters, inputs)

        return preds

    def _make_checkpoint(self, checkpoint_manager, lastest_step=None, *, train_state):
        step = checkpoint_manager.latest_step() if lastest_step is None else lastest_step
        if step is None:
            step = 0
        checkpoint_manager.save(
            step + 1,
            args=ocp.args.StandardSave(train_state),
        )
        return step + 1
    
    def _restore_checkpoint(self, checkpoint_manager, step=None, *, train_state):
        if step is None:
            step = checkpoint_manager.latest_step()
        restored = checkpoint_manager.restore(
            step,
            ocp.args.StandardRestore(train_state),
        )
        
        return restored, step

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
        frozen: dict | None = None,
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

        if self.parameters is None:
            self.initialize(dataset)

        assert self.parameters is not None

        if frozen is None:
            frozen = jax.tree_util.tree_map(lambda _: False, self.parameters)

        optimizers = {True: optax.set_to_zero(), False: self.optimizer}
        tx = optax.multi_transform(
            optimizers,
            frozen,
        )

        train_state = TrainState.create(
            apply_fn=_cached_partial(self.model.apply, **kwargs),
            params=self.parameters,
            tx=tx,
        )

        train_fn = strategy.train_step

        self.reset()

        self.seed, seed = jax.random.split(self.seed)

        for batch in dataset:
            if rng_cols is not None:
                key = jax.random.fold_in(seed, train_state.step)
                keys = jax.random.split(key, len(rng_cols))
                rngs = {name: k for name, k in zip(rng_cols, keys)}
            else:
                rngs = None

            train_state, self.loss_logs, preds = train_fn(
                train_state, self.loss_logs, batch, rngs
            )

            batch_logs = self.compute_loss_log()

            self._params = train_state.params

            op = yield batch_logs
            while op is not None:
                try:
                    op_str, *arg = op
                except:
                    op = yield "Invalid op format."
                else:
                    try:
                        if op_str == "checkpoint":
                            cp_step = self._make_checkpoint(*arg, train_state=train_state)
                            op = yield f"checkpoint - {cp_step}"
                        elif op_str == "restore":
                            train_state, step = self._restore_checkpoint(*arg, train_state=train_state)
                            op = yield f"restored - {step}"
                        elif op_str == "train_state":
                            op = yield train_state
                        else:
                            op = yield f"Unknown operator {op_str}"
                    except Exception as e:
                        op = yield repr(e)

    def save_model(self, path: PathLike, sub_module: Optional[str] = None) -> None:
        """Save the model in a pickled file. The pickle is a tuple of
            (module, weights).

        Args:
            path: The file path.
            sub_module: Optionally only save a sub_module of the model
                by specifying the name
        """
        module = self.model
        params = self.parameters

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

        apply_fn = _cached_partial(self.model.apply, **kwargs)

        predict_fn = strategy.predict

        for data in dataset:
            # inputs, labels, _ = unpack_x_y_sample_weight(data)
            inputs = data[0] if isinstance(data, tuple) else data
            preds = predict_fn(apply_fn, self.parameters, inputs)
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
    def parameters(self) -> Params:
        return self._params

    @parameters.setter
    def parameters(self, new_params: Params | None) -> None:
        if new_params is not None:
            self._params = freeze(new_params)
        else:
            self._params = None
