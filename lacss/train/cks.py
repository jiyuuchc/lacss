from __future__ import annotations

from copy import copy
from functools import partial
from typing import Any, Optional, Sequence

import flax.linen as nn
import jax
import ml_collections
import optax
from xtrain.base_trainer import TrainIterator
from xtrain.strategy import LossLog, TrainState, VMapped
from xtrain.utils import Inputs, unpack_prediction_and_state, unpack_x_y_sample_weight

from lacss.losses import (
    aux_size_loss,
    cks_boundry_loss,
    collaborator_segm_loss,
    instance_overlap_loss,
    self_supervised_instance_loss,
)

from ..modules import ConvNeXt, Lacss, UNet
from ..ops import gradient_reversal
from ..typing import Array, ArrayLike, DataDict, Optimizer
from ..utils import deep_update
from .train import train_fn

jnp = jax.numpy


class LacssCollaborator(nn.Module):
    """Collaborator module for semi-supervised Lacss training

    Attributes:
        conv_spec: conv-net specificaiton for cell border predicition
        unet_spec: specification for unet, used to predict cell foreground
        patch_size: patch size for the unet
        n_cls: number of classes (cell types) of input images
    """

    conv_spec: Sequence[int] = (32, 32)
    unet_spec: Sequence[int] = (16, 32, 64)

    @nn.compact
    def __call__(self, image: ArrayLike, *args, **kwargs) -> DataDict:
        x = jnp.asarray(image)

        net = UNet(self.unet_spec)
        unet_out = net(x)

        x = unet_out[0]
        foreground = nn.Conv(1, (3, 3))(x).squeeze(-1)

        if foreground.shape != x.shape[:-1]:
            foreground = jax.image.resize(foreground, x.shape[:-1], "linear")

        x = image
        for n_features in self.conv_spec:
            x = nn.Conv(n_features, (3, 3), use_bias=False)(x)
            x = nn.LayerNorm(use_scale=False)(x)
            x = jax.nn.relu(x)

        boundary = nn.Conv(2, (3, 3))(x)

        return dict(
            fg_pred=foreground,
            edge_pred=boundary,
        )


class CKSModel(nn.Module):
    principal: Lacss
    collaborator: LacssCollaborator
    config: ml_collections.ConfigDict

    def __call__(self, **kwargs):
        inputs = Inputs()
        inputs = inputs.update(**kwargs)
        outputs = Inputs.apply(
            train_fn,
            self.principal,
            config=self.config,
        )(inputs)

        outputs["predictions"].update(Inputs.apply(self.collaborator)(inputs))

        sigma = self.config.get("sigma", 15.0)
        pi = self.config.get("pi", 2.0)
        w = self.config.get("w", 1e-3)

        outputs["losses"]["cks_loss"] = (
            collaborator_segm_loss(kwargs, outputs, sigma=sigma, pi=pi)
            + cks_boundry_loss(kwargs, outputs)
            + self_supervised_instance_loss(kwargs, outputs)
            + instance_overlap_loss(kwargs, outputs, soft_label=True)
            + aux_size_loss(kwargs, outputs, weight=w)
        )

        return outputs


def _init_cks(it, model, inputs):
    inputs = Inputs.from_value(inputs)
    rngs = it.rngs.copy()
    rngs["params"] = jax.random.PRNGKey(123)
    cks_params = Inputs.apply(model.collaborator.init, rngs)(inputs)["params"]
    params = dict(
        principal=it.parameters,
        collaborator=cks_params,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=it.ctx.optimizer,
    )
    it_ = copy(it)
    it_.train_state = state
    it_.loss_logs = [
        LossLog("losses/lpn_detection_loss"),
        LossLog("losses/lpn_localization_loss"),
        LossLog("losses/cks_loss"),
    ]
    it_.variables = {}
    it_.has_aux = False
    # it_.frozen = jax.tree_util.tree_map(lambda _:False, it_.parameters)
    # it_.freeze("principal")
    return it_


class CKS(VMapped):
    var_key: str = "cks"
    config: ml_collections.ConfigDict = ml_collections.ConfigDict()

    @classmethod
    def train_step(
        cls,
        train_obj: TrainIterator,
        batch: Any,
    ) -> tuple[Any, TrainIterator]:
        inputs, label, _ = unpack_x_y_sample_weight(batch)

        if not cls.var_key in train_obj.variables:
            model = CKSModel(
                train_obj.ctx.model,
                LacssCollaborator(),
                cls.config,
            )
            train_obj.variables[cls.var_key] = _init_cks(train_obj, model, inputs)

        cks = train_obj.variables.pop(cls.var_key)

        if label is None:

            cks.parameters["principal"] = train_obj.parameters
            cks_grad, (prediction, cks) = jax.grad(cls.loss_fn, has_aux=True)(
                cks.parameters, cks, batch
            )
            cks.train_state = cks.train_state.apply_gradients(grads=cks_grad)
            del cks.train_state.params["principal"]

            grads = cks_grad["principal"]

        else:
            grads, (prediction, train_obj) = jax.grad(cls.loss_fn, has_aux=True)(
                train_obj.parameters,
                train_obj,
                batch,
            )

        train_obj.train_state = train_obj.train_state.apply_gradients(grads=grads)

        train_obj.variables[cls.var_key] = cks

        return prediction, train_obj
