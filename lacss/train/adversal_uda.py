from __future__ import annotations

from copy import copy
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import optax
from xtrain.base_trainer import TrainIterator
from xtrain.strategy import LossLog, TrainState, VMapped
from xtrain.utils import Inputs, unpack_x_y_sample_weight

from ..modules.common import FFN, PositionEmbedding2D
from ..ops import gradient_reversal

jnp = jax.numpy


def center_crop(x):
    h, w, _ = x.shape[-3:]
    return x[..., h // 2 - 12 : h // 2 + 12, w // 2 - 12 : w // 2 + 12, :]


def merge_h_w(x):
    return x.reshape(*x.shape[:-3], x.shape[-2] * x.shape[-3], x.shape[-1])


class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, features):
        pos = merge_h_w(PositionEmbedding2D()(features))

        x = merge_h_w(features)

        for _ in range(2):
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadDotProductAttention(4, deterministic=True)(y + pos)
            x = x + FFN(deterministic=True)(y)

        y = nn.LayerNorm()(x)
        token = self.param(
            "token",
            nn.initializers.normal(stddev=0.02),
            y.shape[:-2] + (1, y.shape[-1]),
        )
        y = nn.MultiHeadDotProductAttention(4, deterministic=True)(token, y, y)

        dsc = nn.Dense(1)(y)
        return dsc


class LacssAdv(nn.Module):
    lacss: nn.Module

    @nn.compact
    def __call__(self, image, gt_locations=None, **kwargs):
        det_feature, seg_feature = gradient_reversal(
            self.lacss.get_features(image, None, deterministic=False)
        )

        det_dsc = Discriminator()(det_feature)

        det_adv_loss = optax.sigmoid_binary_cross_entropy(
            det_dsc,
            jnp.ones_like(det_dsc) if gt_locations is None else jnp.zeros_like(det_dsc),
        ).mean()

        seg_dsc = Discriminator()(det_feature)
        seg_adv_loss = optax.sigmoid_binary_cross_entropy(
            seg_dsc,
            jnp.ones_like(seg_dsc) if gt_locations is None else jnp.zeros_like(seg_dsc),
        ).mean()

        # if gt_locations is None:
        #     detected = self.lacss.detector(det_feature)
        #     locations = (detected['predictions']['locations'][:128] / 4).astype(int)
        #     weights  = jnp.clip(detected['predictions']['scores'][:128] - 0.5, 0, 0.5)
        # else:
        #     locations = (gt_locations[:128]/ 4).astype(int)
        #     weights = jnp.where(locations[:, -1] > 0, 1.0, 0.0)

        # seg_patches, _ = self.lacss.segmentor._get_patch(seg_feature, locations)
        # seg_dsc = Discriminator()(seg_patches).squeeze(axis=(-1, -2))

        # seg_adv_loss = optax.sigmoid_binary_cross_entropy(
        #     seg_dsc,
        #     jnp.ones_like(seg_dsc) if gt_locations is None else jnp.zeros_like(seg_dsc)
        # )
        # seg_adv_loss = (seg_adv_loss * weights).sum() / (weights.sum() + 1e-6)

        output = dict(
            det_adv_loss=det_adv_loss,
            seg_adv_loss=seg_adv_loss,
        )

        return output


def _init_uda_it(it, inputs):
    model = LacssAdv(it.ctx.model)
    rngs = it.rngs.copy()
    rngs["params"] = jax.random.PRNGKey(123)
    params = model.init(rngs, inputs["image"][0])["params"]
    params["lacss"] = it.parameters
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=it.ctx.optimizer,
    )
    it_ = copy(it)
    it_.train_state = state
    it_.loss_logs = [LossLog("det_adv_loss"), LossLog("seg_adv_loss")]
    # it_.loss_logs = [LossLog("det_adv_loss")]
    it_.variables = {}
    it_.has_aux = False
    it_.frozen = jax.tree_util.tree_map(lambda _: False, it_.parameters)
    it_.freeze("lacss")
    return it_


class AdversalUDA(VMapped):
    var_key: str = "adv"
    beta: float = 0.5

    @classmethod
    def train_step(
        cls,
        train_obj: TrainIterator,
        batch: Any,
    ) -> tuple[Any, TrainIterator]:
        inputs, label, _ = unpack_x_y_sample_weight(batch)

        if not cls.var_key in train_obj.variables:
            train_obj.variables[cls.var_key] = _init_uda_it(train_obj, inputs)

        uda_it = train_obj.variables[cls.var_key]
        del train_obj.variables[cls.var_key]

        uda_it.parameters["lacss"] = train_obj.parameters
        uda_grad, (prediction, uda_it) = jax.grad(cls.loss_fn, has_aux=True)(
            uda_it.parameters, uda_it, batch
        )
        uda_it.train_state = uda_it.train_state.apply_gradients(grads=uda_grad)
        del uda_it.train_state.params["lacss"]

        if label is not None:
            grads, (inner_pred, train_obj) = jax.grad(cls.loss_fn, has_aux=True)(
                train_obj.parameters,
                train_obj,
                batch,
            )
            grads = jax.tree_util.tree_map(
                lambda a, b: a * cls.beta + b, uda_grad["lacss"], grads
            )
            prediction.update(inner_pred)
        else:
            grads = jax.tree_util.tree_map(lambda x: x * cls.beta, uda_grad["lacss"])

        train_obj.train_state = train_obj.train_state.apply_gradients(grads=grads)

        train_obj.variables[cls.var_key] = uda_it

        return prediction, train_obj
