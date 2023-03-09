import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState

from ..utils import Inputs
from .pytree import Pytree


class Eager:
    @classmethod
    def loss_fn(cls, params, state, loss_log, inputs, labels, rngs):
        inputs_obj = Inputs.from_value(inputs)
        preds = state.apply_fn(
            {"params": params},
            *inputs_obj.args,
            **inputs_obj.kwargs,
            training=True,
            rngs=rngs,
        )

        args = dict(
            inputs=inputs,
            preds=preds,
            **labels,
        )

        total_loss, _ = loss_log.update(**args)

        return total_loss, (state, loss_log, preds)

    @classmethod
    def predict(cls, state, inputs):  # only if model is immutable
        # print("JIT Predict")
        inputs_obj = Inputs.from_value(inputs)
        preds = state.apply_fn(
            {"params": state.params}, *inputs_obj.args, **inputs_obj.kwargs
        )
        return preds

    @classmethod
    def train_step(
        cls,
        state: TrainState,
        loss_log: Pytree,
        inputs: tp.Any,
        labels: tp.Any,
        rngs: tp.Optional[dict],
    ) -> tp.Tuple[TrainState, Pytree, tp.Any]:
        # print('JIT train_step')

        if labels is None:
            labels = {}
        elif not isinstance(labels, dict):
            labels = dict(target=labels)

        grads, (state, loss_log, preds) = jax.grad(cls.loss_fn, has_aux=True)(
            state.params,
            state,
            loss_log,
            inputs,
            labels,
            rngs,
        )
        state = state.apply_gradients(grads=grads)

        return state, loss_log, preds


class Core(Eager):
    predict = jax.jit(Eager.predict)


class JIT(Core):
    train_step = jax.jit(Core.train_step)


class Distributed(Eager):
    @classmethod
    def _train_step(
        cls,
        state: TrainState,
        loss_log: Pytree,
        inputs: tp.Any,
        labels: tp.Any,
        rngs: tp.Optional[dict],
    ) -> tp.Tuple[TrainState, Pytree, tp.Any]:
        # print("JITTTTING")
        axis_index = jax.lax.axis_index("device")
        rng = jax.tree_util.tree_map(
            lambda key: jax.random.fold_in(key, axis_index), rng
        )

        if labels is None:
            labels = {}
        elif not isinstance(labels, dict):
            labels = dict(target=labels)

        grads, (state, loss_log, preds) = jax.grad(cls.loss_fn, has_aux=True)(
            state.params,
            state,
            loss_log,
            inputs,
            labels,
            rngs,
        )

        grads = jax.lax.pmean(grads, axis_name="device")
        state = state.apply_gradients(grads=grads)

        # aggregate logs
        loss_log = jax.tree_map(partial(jax.lax.pmean, axis_name="device"), loss_log)

        #         # sync batch statistics
        #         model.map(partial(jax.lax.pmean, axis_name="device"), tx.BatchStat, inplace=True)

        return state, loss_log, preds

    predict = jax.pmap(
        Eager.predict,
        in_axes=(None, 0),
    )


Distributed.train_step = jax.pmap(
    Distributed._train_step,
    in_axes=(None, None, 0, 0, None),
    out_axes=(None, None, 0),
)
