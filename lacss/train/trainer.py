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
from .loss import LossLog

# so that multiple calls return the same obj
# this avoids JIT when supplying partial func as args
_cached_partial = lru_cache(partial)

LOSSES = tp.Union[tp.Callable, tp.Sequence[tp.Callable]]


def _get_iterator(g):
    try:
        it = iter(g)
    except:
        it = iter(g())

    return it


class Trainer:
    """ FLAX trainer
    The design is to be minimal but help avoiding certain biolerplate code when train with FLAX

    Attributes:
        model: A Flax module
        losses: A list of loss function (or other callabels). 
        optimizer: An optax optimizer
        seed: RNG seed
        strategy: Training strategy. See lacss.train.strategy module. 
        params: Current model parameters. This is a frozen dict. This is read/write.
        initialized: Whether the model has been initialized with an optimizer and initial weights.
    
    Constructor:
        model: A Flax module
        losses: A loss function (callabels) or a list of loss functions, or None. Default is None.
            These should have the call signiture of: 
                loss = loss_fn(inputs, labels, preds)
            The "preds" is the model output. The "inputs" and "labels" are from the train dataset.
            The model trains on the sum of all losses.
        optimizer: An optax optimzier or None
        seed: integer RNG seed. Default is 42.
        strategy: Training strategy. See lacss.train.strategy module. Default is JIT, which trains
            on a single GPU with unbatched input data. Use VMapped strategy for bathced input. Use
            Distributed for multi-GPU training. Use Eager for debugging (no JIT). You can supply 
            your own stategy class.
    """
    def __init__(
        self,
        model: nn.Module,
        losses: tp.Optional[LOSSES] = None,
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
        """ Reset internal loss value tracking

        Args:
            loss_weights: Optional weights of individual loss functions. If not None, the
                total loss is weighted sum.
        """
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

    def initialize(self, dataset, tx: GradientTransformation = None):
        """ Initialize the model weights and optimizer states.

        Args:
            dataset: An iterator or generator function to supply the training dataset.
                The dataset should yield (inputs, labels) or inputs. In the latter case
                the labels = None. The inputs is either a tuple or a dict. If the inputs 
                is a dict, the keys are interpreted as the names for keyword args of the 
                model's __call__ function.
            tx: Optional optax optimzier for when the object was constructed without an 
                optimizer
        """
        if tx is None:
            tx = self.optimizer

        if not self._initialized:
            if self.model.scope is None:
                peek = next(_get_iterator(dataset))
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
        """ Calling the underlying model

        Args:
            inputs: A tuple or a dict as the inputs for the model.
            strategy: Optionally supply a differnt calling strategy as the default one
                supplied at the constructor.
            **kwargs: Additional keyword args passed on to the model
        
        Returns:
            Model outputs.
        """
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
        """ Create the training iterator

        Args:
            dataset: An iterator or generator function to supply the training dataset.
                See initialize()
            strategy: Optionally override the default strategy.
            rng_cols: Names of any RNG used by the model. Should be a list of strs.
            **kwargs: Additional keyward args passed to the model. E.g. "training=True"
        
        Returns:
            A iterator. Stepping through the iterator will train the model. The iterator
                itself returns a loss log dict, which are mean loss values for each loss
                function.
        
        Usage example:
            trainer.reset()
            train_it = trainer.train(train_dataset, training=True)
            for k in range(train_steps):
                loss_logs = next(train_it)
                if k % 1000 == 0:
                    print(loss_logs)
        """
        if strategy is None:
            strategy = self._strategy

        if not self._initialized:
            raise ValueError("Try to run uninitialized trainer")

        self.reset()

        self.seed, seed = jax.random.split(self.seed)
        for step, data in enumerate(_get_iterator(dataset)):
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
        """ Save the model in a pickled file. The pickle is a tuple of
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
            path = pathlib.Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            cloudpickle.dump((module, params), f)

    def checkpoint(self, path):
        """ Make a checkpoint of the trainer. This saves the model as well as
        the training states.

        Args:
            path: The file path.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def from_checkpoint(path):
        """ Restore from the checkpoint.

        Args:
            path: The checkpoint file path.

        Returns:
            A new trainer object.
        """
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
        """ Create test/validation iterator.

        Args:
            dataset: An iterator or generator function to supply the testing data.
                The iterator should yield a tupple of (inputs, labels). The labels
                should be a dict.
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

        # self.initialize(dataset, strategy)

        for data in _get_iterator(dataset):
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
        """ A convient function to compute all metrics.

        Args:
            same as the test() fucntion
        
        Returns:
            A metric dict. Keys are metric names.
        """
        for metrics in self.test(*args, **kwargs):
            pass
        return {_get_name(m): m.compute() for m in metrics}

    @property
    def params(self):
        return self.state.params

    @params.setter
    def params(self, new_params):
        self.state = self.state.replace(params=new_params)

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
