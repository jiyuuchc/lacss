from typing import Dict, List, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from .common import *


class AuxInstanceEdge(nn.Module):
    conv_spec: Sequence[int] = (24, 64, 64)
    n_groups: int = 1
    share_weights: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, *, category: Optional[int] = None
    ) -> jnp.ndarray:
        for n in self.conv_spec:
            x = nn.Conv(n, (3, 3), use_bias=False)(x)
            x = nn.GroupNorm(num_groups=None, group_size=1, use_scale=False)(
                x[None, ...]
            )[0]
            x = jax.nn.relu(x)

        x = nn.Conv(self.n_groups, (3, 3))(x)
        x = x[..., category if category is not None else 0]

        # x = jax.nn.sigmoid(x)

        return x


class AuxForeground(nn.Module):
    conv_spec: Sequence[int] = (24, 64, 64)
    n_groups: int = 1
    share_weights: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, *, category: Optional[int] = None
    ) -> jnp.ndarray:
        orig = x
        n_ch = self.conv_spec[0]
        x = nn.Conv(n_ch, (3, 3), strides=(2, 2), use_bias=False)(x)
        x = nn.GroupNorm(num_groups=n_ch, use_scale=False)(x[None, ...])[0]
        x = jax.nn.relu(x)
        for n_ch in self.conv_spec[1:]:
            x = nn.Conv(n_ch, (3, 3), use_bias=False)(x)
            x = nn.GroupNorm(num_groups=n_ch, use_scale=False)(x[None, ...])[0]
            x = jax.nn.relu(x)

        x = SpatialAttention()(x)
        x = jax.image.resize(x, orig.shape[:-1] + x.shape[-1:], "cubic")
        x = jnp.concatenate([orig, x], axis=-1)

        x = nn.Conv(self.n_groups, (3, 3))(x)
        x = x[..., category if category is not None else 0]

        # x = jax.nn.sigmoid(x)

        return x
