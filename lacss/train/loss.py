from __future__ import annotations

from typing import Sequence, Union

import jax.numpy as jnp
from flax import struct

from .utils import _get_name


class LossLog(struct.PyTreeNode):
    loss_fn: LossFunc = struct.field(pytree_node=False)
    weight: jnp.ndarray = 1.0
    cnt: jnp.ndarray = 0.0
    sum: jnp.ndarray = 0.0

    def update(self, **kwargs):
        loss = self.loss_fn(**kwargs) * self.weight
        new_log = self.replace(cnt=self.cnt + 1, sum=self.sum + loss)
        return loss, new_log

    def compute(self):
        return self.sum / self.cnt
