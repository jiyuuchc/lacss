import typing as tp
from functools import partial
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ..utils import Inputs
from .wrapper import WrappedGT, WrappedModule
from .pytree import Pytree

class Eager:
    @classmethod
    def predict(cls, trainer, inputs):  # only if model is immutable
        print("JIT Predict")
        inputs_obj = Inputs.from_value(inputs)
        preds = trainer.model(*inputs_obj.args, **inputs_obj.kwargs) 
        return preds

    @classmethod
    def _loss_fn(
        cls, params,
        trainer: Pytree,
        inputs: tp.Any,
        labels: tp.Any,
        rngs: tp.Optional[dict],
    ) -> tp.Tuple[jnp.ndarray, tp.Tuple[Pytree, dict, tp.Any]]:
        trainer.model.update_variables(dict(params = params)) # so the compiler can track the effect of params

        # don't call predict() because model might be mutable
        inputs_obj = Inputs.from_value(inputs)
        preds = trainer.model(*inputs_obj.args, **inputs_obj.kwargs, training=True, rngs=rngs)

        args = dict(
            inputs = inputs,
            preds = preds,
            **labels,
        )

        total_loss, _ = trainer.loss_log.update(**args)

        return total_loss, (trainer, preds)

    @classmethod
    def init_step(
        cls,
        trainer: Pytree,
        inputs: tp.Any,
    ) -> Pytree:
        # print('JIT init_step')
        if not trainer.model.initialized:
            trainer.seed, key = jax.random.split(trainer.seed)
            inputs_obj = Inputs.from_value(inputs)
            trainer.model.init(key, *inputs_obj.args, **inputs_obj.kwargs)
        if trainer.optimizer is not None and not trainer.optimizer.initialized:
            trainer.optimizer.init(trainer.model.get_variables())

        return trainer

    @classmethod
    def train_step(
        cls,
        trainer: Pytree,
        inputs: tp.Any,
        labels: tp.Any,
        rngs: tp.Optional[dict],
    ) -> tp.Tuple[Pytree, tp.Any]:
        # print('JIT train_step')

        params = trainer.model.get_variables()

        if labels is None:
            labels = {}
        elif not isinstance(labels, dict):
            labels = dict(target = labels)

        grads, (trainer, preds) = jax.grad(cls._loss_fn, has_aux=True)(
            params, trainer, inputs, labels, rngs,
        )
        params = trainer.optimizer.update(grads, params)
        trainer.model.update_variables(dict(params = params))

        return trainer, preds

class Core(Eager):
    predict = jax.jit(Eager.predict)

class JIT(Core):
    train_step = jax.jit(Core.train_step)
    # test_step = jax.jit(Core.test_step)

# class Distributed(Core):
#     @classmethod
#     def _train_step(
#         cls,
#         model: Model,
#         optimizer: tx.Optimizer,
#         loss_logs: tx.LossAndLogs,
#         inputs: tp.Any,
#         labels: tp.Any,
#     ):
#         print("JITTTTING")
#         # copies of models have differnt rng key
#         axis_index = jax.lax.axis_index("device")
#         model.map(lambda key: jax.random.split(key, jax.device_count())[axis_index], tx.Rng, inplace=True)

#         params = model.trainable_parameters()
#         batch_loss_logs = tx.copy(loss_logs)
#         grads, (model, batch_loss_logs, logs) = jax.grad(cls._loss_fn, has_aux=True)(
#             params, model.train(), batch_loss_logs, inputs, labels
#         )

#         grads = jax.lax.pmean(grads, axis_name="device")
#         params = optimizer.update(grads, params)
#         model = model.merge(params)

#         # sync batch statistics
#         model.map(partial(jax.lax.pmean, axis_name="device"), tx.BatchStat, inplace=True)

#         # update metrics from batch_loss_logs
#         batch_loss_logs = jax.tree_map(lambda x,y: y-x, loss_logs.MetricState(), batch_loss_logs.MetricState())
#         batch_loss_logs.map(partial(jax.lax.psum, axis_name='device'), tx.MetricState, inplace=True)
#         loss_logs = loss_logs.merge(jax.tree_map(lambda x,y: x+y, loss_logs.MetricState(), batch_loss_logs.MetricState()))

#         # aggregate logs
#         logs = jax.tree_map(partial(jax.lax.pmean, axis_name='device'), logs)

#         return model, optimizer, loss_logs, logs

#     @classmethod
#     def _test_step(
#         cls,
#         model: Model,
#         metrics: tx.Metric,
#         inputs: tp.Any,
#         labels: tp.Any,
#     ):
#         preds = cls.predict(inputs)

#         labels, preds = jax.lax.all_gather((labels, preds), axis_name='device')
#         labels, preds = jax.tree_map(
#             lambda v: v.reshape([-1]+v.shape[2:]),
#             (labels, preds),
#         )

#         metrics.update(
#             preds=preds,
#             model=model,
#             inputs=inputs,
#             **labels,
#         )

#         return metrics

# Distributed.train_step = jax.pmap(
#     Distributed._train_step,
#     in_axes =(None, None, None, 0, 0),
#     out_axes=None, 
#     axis_name="device"
#     )

# Distributed.test_step = jax.pmap(
#     Distributed._test_step,
#     in_axes = (None, None, 0, 0),
#     out_axes = None,
#     axis_name="device"
#     )
