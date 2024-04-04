from __future__ import annotations

from typing import Sequence, Union

import jax.numpy as jnp
from flax import struct

from ..typing import LossFunc
from .utils import _get_name


class LossLog(struct.PyTreeNode):
    loss_fn: LossFunc = struct.field(pytree_node=False)
    weight: float = 1.0
    cnt: float = 0.0
    sum: float = 0.0

    # update() is meant to be called in JAX transformation so it cannot
    # modify its fields. Instead it returns a new copy of self.
    def update(self, batch, prediction):
        if isinstance(self.loss_fn, str):
            loss = prediction
            for k in self.loss_fn.split("/"):
                loss = loss[k]
        else:
            loss = self.loss_fn(batch, prediction)

        if loss is None:
            return 0.0, self

        loss *= self.weight

        return loss, self.replace(cnt=self.cnt + 1, sum=self.sum + loss)

    def compute(self):
        return self.sum / self.cnt

    def reset(self):
        object.__setattr__(self, "cnt", 0.0)
        object.__setattr__(self, "sum", 0.0)

    def __repr__(self) -> str:
        return _get_name(self.loss_fn) + f": {float(self.sum / self.cnt):.4f}"
