from functools import partial
from dataclasses import dataclass
import pathlib
import cloudpickle
import pickle
import typing as tp
import treex as tx
from optax import GradientTransformation
import jax
jnp = jax.numpy

from .data import *
from .strategy import *

''' based on treex examples
'''

Model = tx.Module
Logs = tp.Dict[str, tp.Any]

class Trainer():
    def __init__(
        self,
        model: tx.Module,
        optimizer: tp.Optional[tp.Union[tx.Optimizer, GradientTransformation]] = None,
        losses: tp.Any = None,
        metrics: tp.Any = None,
        seed: int = 42,
        train_strategy = JIT,
    ):
        self.model = model
        self.seed = seed
        
        if optimizer is not None:
            self.optimizer = optimizer \
                if isinstance(optimizer, tx.Optimizer) \
                else tx.Optimizer(optimizer)
        else:
            self.optimizer = None
        
        self.losses = losses
        self.metrics = metrics
        
        self._strategy = train_strategy
        self.loss_and_logs = None
        self._initialized = False

    def reset_metrics(self):
        if self.loss_and_logs is not None:
            self.loss_and_logs.reset()

    def initialize(self, dataset, strategy=None):
        if strategy is None:
            strategy = self._strategy

        peek = next(dataset())
        inputs, _, _ = unpack_x_y_sample_weight(peek)
        init_fn = strategy.init_step
        self.model, self.optimizer = init_fn(
            self.seed,
            self.model,
            self.optimizer,
            inputs,
        )

        if self.loss_and_logs is None:
            self.loss_and_logs = tx.LossAndLogs(
                losses=self.losses,
                metrics=self.metrics,
                aux_losses=self.model.loss_logs(),
                aux_metrics=self.model.metric_logs(),
            )

        self._initialized = True

    def __call__(self, dataset, strategy=None):
        if strategy is None:
            strategy = self._strategy

        if not self._initialized:
            self.initialize(dataset, strategy)

        self.reset_metrics()

        for data in dataset():
            inputs, labels, _ = unpack_x_y_sample_weight(data)
            train_fn = strategy.train_step
            self.model, self.optimizer, self.loss_and_logs, batch_logs = train_fn(
                self.model.train(),
                self.optimizer,
                self.loss_and_logs,
                inputs,
                labels,
            )

            yield batch_logs

    def checkpoint(self, path):
        # model = self.local()

        if isinstance(path, str):
            path = pathlib.Path(path)

        path.mkdir(parents=True, exist_ok=True)

        with open(path / "trainer.pkl", "wb") as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def from_checkpoint(path):
        if isinstance(path, str):
            path = pathlib.Path(path)

        try:
            _bytes = (path / "trainer.pkl").read_bytes()
        except BaseException as e:
            raise OSError(f"Could not load the checkpoint. Got exception: {e}")

        return pickle.loads(_bytes)

    def test(self, dataset, metrics=None, strategy=None):
        if metrics is None:
            metrics = self.loss_and_logs
        
        if strategy is None:
            strategy = self._strategy

        metrics.reset()

        for data in dataset():
            inputs, labels, _ = unpack_x_y_sample_weight(data)
            test_fn = strategy.test_step
            metrics = test_fn(self.model.eval(), metrics, inputs, labels)

            yield metrics
        
    def test_and_compute(self, *args, **kwargs):
        for metrics in self.test(*args, **kwargs):
            pass
        return metrics.compute()
    
