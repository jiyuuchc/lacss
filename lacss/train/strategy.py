from __future__ import annotations

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training.train_state import TrainState

from .loss import LossLog
from .utils import Inputs


class Eager:
    @classmethod
    def loss_fn(cls, params, state, loss_logs, batch, rngs):
        # inputs, labels, sample_weight = batch
        inputs = batch[0] if isinstance(batch, tuple) else batch
        # if sample_weight is None:
        #     sample_weight = 1.0

        inputs_obj = Inputs.from_value(inputs)

        preds = state.apply_fn(
            {"params": params},
            *inputs_obj.args,
            **inputs_obj.kwargs,
            rngs=rngs,
        )

        args = dict(
            batch=batch,
            prediction=preds,
        )

        losses, loss_logs = zip(*[loss_log.update(**args) for loss_log in loss_logs])
        total_loss = sum(losses)

        return total_loss, (state, loss_logs, preds)

    @classmethod
    def init_fn(cls, key, model, inputs):
        inputs_obj = Inputs.from_value(inputs)

        state = model.init(key, *inputs_obj.args, **inputs_obj.kwargs)

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
        loss_logs: tp.Sequence[LossLog],
        batch: tp.Any,
        rngs: tp.Optional[dict],
    ) -> tp.Tuple[TrainState, tp.Sequence[LossLog], tp.Any]:
        # print('JIT train_step')
        grads, (state, loss_logs, preds) = jax.grad(cls.loss_fn, has_aux=True)(
            state.params,
            state,
            loss_logs,
            batch,
            rngs,
        )
        state = state.apply_gradients(grads=grads)

        return state, loss_logs, preds


class Core(Eager):
    predict = jax.jit(Eager.predict)

    @classmethod
    def init_fn(cls, key, model, inputs):
        inputs_obj = Inputs.from_value(inputs)

        state = jax.jit(model.init)(key, *inputs_obj.args, **inputs_obj.kwargs)

        return state


class JIT(Core):
    train_step = jax.jit(Core.train_step)


class _Distributed(Eager):
    @classmethod
    def _train_step(
        cls,
        state: TrainState,
        loss_logs: tp.Sequence[LossLog],
        batch: tp.Any,
        rngs: tp.Optional[dict],
    ) -> tp.Tuple[TrainState, tp.Sequence[LossLog], tp.Any]:
        # print("JITTTTING")
        axis_index = jax.lax.axis_index("mapped")
        rngs = jax.tree_util.tree_map(
            lambda key: jax.random.fold_in(key, axis_index), rngs
        )

        grads, (state, loss_logs, preds) = jax.grad(cls.loss_fn, has_aux=True)(
            state.params,
            state,
            loss_logs,
            batch,
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
    def init_fn(cls, key, model, inputs):
        inputs = jax.tree_map(lambda v: v[0], inputs)
        return Eager.init_fn(key, model, inputs)


class Distributed(_Distributed):
    train_step = jax.pmap(
        _Distributed._train_step,
        axis_name="mapped",
        in_axes=(None, None, 0, None),
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
            in_axes=(None, None, 0, None),
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
