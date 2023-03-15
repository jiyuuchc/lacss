import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState

from ..utils import Inputs
from .loss import Loss, LossLog


class Eager:
    @classmethod
    def loss_fn(cls, params, state, loss_logs, inputs, labels, rngs):
        inputs_obj = Inputs.from_value(inputs)
        preds = state.apply_fn(
            {"params": params},
            *inputs_obj.args,
            **inputs_obj.kwargs,
            rngs=rngs,
        )

        args = dict(
            inputs=inputs,
            preds=preds,
            **labels,
        )

        losses, loss_logs = zip(*[loss_fn.update(**args) for loss_fn in loss_logs])
        total_loss = sum(losses)

        return total_loss, (state, loss_logs, preds)

    @classmethod
    def init_fn(cls, key, model, inputs, tx):
        inputs_obj = Inputs.from_value(inputs)
        variables = model.init(key, *inputs_obj.args, **inputs_obj.kwargs)
        state = TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=tx,
        )
        return state

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
        loss_logs: tp.Dict[str, LossLog],
        inputs: tp.Any,
        labels: tp.Any,
        rngs: tp.Optional[dict],
    ) -> tp.Tuple[TrainState, dict, tp.Any]:
        # print('JIT train_step')

        if labels is None:
            labels = {}
        elif not isinstance(labels, dict):
            labels = dict(target=labels)

        grads, (state, loss_logs, preds) = jax.grad(cls.loss_fn, has_aux=True)(
            state.params,
            state,
            loss_logs,
            inputs,
            labels,
            rngs,
        )
        state = state.apply_gradients(grads=grads)

        return state, loss_logs, preds


class Core(Eager):
    predict = jax.jit(Eager.predict)


class JIT(Core):
    train_step = jax.jit(Core.train_step)


class _Distributed(Eager):
    @classmethod
    def _train_step(
        cls,
        state: TrainState,
        loss_logs: tp.Dict[str, LossLog],
        inputs: tp.Any,
        labels: tp.Any,
        rngs: tp.Optional[dict],
    ) -> tp.Tuple[TrainState, tp.Dict[str, LossLog], tp.Any]:
        # print("JITTTTING")
        axis_index = jax.lax.axis_index("mapped")
        rngs = jax.tree_util.tree_map(
            lambda key: jax.random.fold_in(key, axis_index), rngs
        )

        if labels is None:
            labels = {}
        elif not isinstance(labels, dict):
            labels = dict(target=labels)

        grads, (state, loss_logs, preds) = jax.grad(cls.loss_fn, has_aux=True)(
            state.params,
            state,
            loss_logs,
            inputs,
            labels,
            rngs,
        )

        grads = jax.lax.pmean(grads, axis_name="mapped")
        state = state.apply_gradients(grads=grads)

        # aggregate logs
        loss_logs = jax.tree_map(partial(jax.lax.pmean, axis_name="mapped"), loss_logs)

        #         # sync batch statistics
        #         model.map(partial(jax.lax.pmean, axis_name="mapped"), tx.BatchStat, inplace=True)

        return state, loss_logs, preds

    @classmethod
    def init_fn(cls, key, model, inputs, tx):
        inputs = jax.tree_map(lambda v: v[0], inputs)
        return Eager.init_fn(key, model, inputs, tx)


class Distributed(_Distributed):
    train_step = jax.pmap(
        _Distributed._train_step,
        axis_name="mapped",
        in_axes=(None, None, 0, 0, None),
        out_axes=(None, None, 0),
    )

    predict = jax.pmap(
        Eager.predict,
        axis_name="mapped",
        in_axes=(None, 0),
    )


class VMapped(_Distributed):
    train_step = jax.jit(
        jax.vmap(
            _Distributed._train_step,
            axis_name="mapped",
            in_axes=(None, None, 0, 0, None),
            out_axes=(None, None, 0),
        )
    )

    predict = jax.jit(
        jax.vmap(
            Eager.predict,
            axis_name="mapped",
            in_axes=(None, 0),
        )
    )
