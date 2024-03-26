from __future__ import annotations

from typing import Sequence, Union

import jax.numpy as jnp
from flax import struct

from ..typing import LossFunc
from .utils import _get_name


class LossLog(struct.PyTreeNode):
    loss_fn: LossFunc = struct.field(pytree_node=False)
    reduction: str = struct.field(pytree_node=False, default="mean")
    weight: float = 1.0
    cnt: float = 0.0
    sum: float = 0.0

    def update(self, batch, prediction):
        if isinstance(self.loss_fn, str):
            loss = prediction
            for k in self.loss_fn.split("/"):
                loss = loss[k]
        else:
            loss = self.loss_fn(batch, prediction)
        if loss is None:
            return 0.0, self

        if self.reduction == "mean":
            loss = jnp.mean(jnp.asarray(loss))
        elif self.reduction == "sum":
            loss = jnp.sum(jnp.asarray(loss))
        else:
            raise ValueError(f"unknown reduction {self.reduction}")

        loss *= self.weight
        return loss, self.replace(cnt=self.cnt + 1, sum=self.sum + loss)

    def compute(self):
        return self.sum / self.cnt

    def reset(self) -> LossLog:
        return self.replace(
            cnt=0,
            sum=0,
        )

    def __repr__(self) -> str:
        return _get_name(self.loss_fn) + f": {float(self.sum / self.cnt):.4f}"
