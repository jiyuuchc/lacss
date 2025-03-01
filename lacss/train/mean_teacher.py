from __future__ import annotations

from functools import partial
from typing import Any

import flax.linen as nn
import jax
import optax
from xtrain.base_trainer import TrainIterator
from xtrain.strategy import LossLog, TrainState, VMapped
from xtrain.utils import Inputs, unpack_prediction_and_state, unpack_x_y_sample_weight

from ..utils import deep_update

jnp = jax.numpy


def lacss_mt_train_fn(module, image, teacher_prediction):
    n_train_instances = 16

    seg_locs = teacher_prediction["predictions"]["locations"][:n_train_instances]

    cutout_locs = (
        jax.random.uniform(module.make_rng("dropout"), shape=seg_locs.shape) * 50
        - 25
        + seg_locs
    )
    cutout_locs = jnp.where(seg_locs >= 0, cutout_locs.astype("int32"), 9999)
    image, _ = jax.lax.scan(
        lambda img, loc: (
            jax.lax.dynamic_update_slice(
                img,
                jnp.zeros([20, 20, 3], img.dtype),
                (loc[0] - 10, loc[1] - 10, 0),
            ),
            None,
        ),
        image,
        cutout_locs,
    )

    x_det, x_seg = module.get_features(image, None, deterministic=False)
    outputs = module.detector(x_det)

    pred_cls_prob = jax.nn.sigmoid(outputs["detector"]["logits"])
    gt_cls_prob = jax.nn.sigmoid(teacher_prediction["detector"]["logits"])
    mt_det_loss = pred_cls_prob * (1 - gt_cls_prob) + (1 - pred_cls_prob) * gt_cls_prob
    mt_det_loss = mt_det_loss * jnp.abs(gt_cls_prob - 0.5)
    mt_det_loss = mt_det_loss.sum() / jnp.abs(gt_cls_prob - 0.5).sum()

    pred_regr = outputs["detector"]["regressions"]
    gt_regr = teacher_prediction["detector"]["regressions"]
    mt_loc_loss = optax.l2_loss(pred_regr / 4, gt_regr / 4) * gt_cls_prob[:, None]
    mt_loc_loss = mt_loc_loss.sum() / gt_cls_prob[:, None].sum()

    segmentor_out = module.segmentor(x_seg, seg_locs)
    outputs = deep_update(outputs, segmentor_out)

    pred_segmentations = jax.nn.sigmoid(segmentor_out["predictions"]["segmentations"])
    gt_segmentations = jax.nn.sigmoid(
        teacher_prediction["predictions"]["segmentations"][:n_train_instances]
    )
    # mt_seg_loss = pred_segmentations * (1 - gt_segmentations) + (1 - pred_segmentations) * gt_segmentations
    mt_seg_loss = optax.l2_loss(pred_segmentations, gt_segmentations) * jnp.abs(
        gt_segmentations - 0.5
    )
    mt_seg_loss = mt_seg_loss.sum() / jnp.abs(gt_segmentations - 0.5).sum()

    outputs["losses"] = dict(
        mt_det_loss=mt_det_loss,
        mt_loc_loss=mt_loc_loss,
        mt_seg_loss=mt_seg_loss,
    )

    return outputs


def lacss_mt_loss(params, teacher_params, train_obj, inputs):
    inputs = Inputs.from_value(inputs)
    batch_size = jax.tree_util.tree_leaves(inputs)[0].shape[0]

    inputs = inputs.update(
        rngs={
            name: jax.random.split(jax.random.fold_in(rng, train_obj.step), batch_size)
            for name, rng in train_obj.rngs.items()
        },
        teacher_prediction=jax.vmap(
            Inputs.apply(
                train_obj.ctx.model.apply,
                dict(params=teacher_params),
            )
        )(inputs),
    )

    model_out = jax.vmap(
        Inputs.apply(
            nn.apply(lacss_mt_train_fn, train_obj.ctx.model),
            dict(params=params),
        )
    )(inputs)

    total_loss = sum(
        [
            model_out["losses"]["mt_det_loss"].sum(),
            model_out["losses"]["mt_seg_loss"].sum(),
            model_out["losses"]["mt_loc_loss"].sum(),
        ]
    )

    return total_loss, model_out


class MeanTeacher(VMapped):
    alpha: float = 0.05
    var_key: str = "teacher"

    @classmethod
    def train_step(
        cls,
        train_obj: TrainIterator,
        batch: Any,
    ) -> tuple[Any, Any]:
        inputs, label, _ = unpack_x_y_sample_weight(batch)

        if not cls.var_key in train_obj.variables:
            train_obj.variables[cls.var_key] = dict(
                params=train_obj.parameters,
                mt_loss_log=[
                    LossLog("losses/mt_det_loss"),
                    LossLog("losses/mt_seg_loss"),
                    LossLog("losses/mt_loc_loss"),
                ],
            )

        teacher_vars = train_obj.variables.pop(cls.var_key)

        if label is not None:
            prediction, train_obj = VMapped.train_step(train_obj, batch)

        else:
            grads, prediction = jax.grad(lacss_mt_loss, has_aux=True)(
                train_obj.parameters, teacher_vars["params"], train_obj, inputs
            )
            train_obj.train_state = train_obj.train_state.apply_gradients(grads=grads)
            for mt_losslog in teacher_vars["mt_loss_log"]:
                mt_losslog.update(batch, prediction)

        # update teacher params
        teacher_vars["params"] = jax.tree_map(
            lambda a, b: a * (1 - cls.alpha) + b * cls.alpha,
            teacher_vars["params"],
            train_obj.parameters,
        )

        train_obj.variables[cls.var_key] = teacher_vars

        return prediction, train_obj
