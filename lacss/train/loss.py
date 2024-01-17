from __future__ import annotations

from typing import Sequence, Union

import jax.numpy as jnp
from flax import struct

from ..typing import LossFunc
from .utils import _get_name


class LossLog(struct.PyTreeNode):
    loss_fn: LossFunc = struct.field(pytree_node=False)
    weight: jnp.ndarray = 1.0
    cnt: jnp.ndarray = 0.0
    sum: jnp.ndarray = 0.0

    def update(self, **kwargs):
        loss = self.loss_fn(**kwargs)
        if loss is None:
            return 0.0, self
        else:
            loss *= self.weight
            return loss, self.replace(cnt=self.cnt + 1, sum=self.sum + loss)

    def compute(self):
        return self.sum / self.cnt
