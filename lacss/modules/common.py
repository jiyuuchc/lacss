from __future__ import annotations

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..typing import Array, ArrayLike


class ChannelAttention(nn.Module):
    squeeze_factor: int = 16

    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
    ) -> Array:

        orig_shape = x.shape

        x = x.reshape(-1, orig_shape[-1])
        x_backup = x

        x = jnp.stack([x.max(axis=0), x.mean(axis=0)])

        se_channels = orig_shape[-1] // self.squeeze_factor
        x = nn.Dense(se_channels)(x)
        x = jax.nn.relu(x)
        x = x[0] + x[1]

        x = nn.Dense(orig_shape[-1])(x)
        x = jax.nn.sigmoid(x)

        x = x_backup * x
        x = x.reshape(orig_shape)

        return x


class SpatialAttention(nn.Module):
    filter_size: int = 7

    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
    ) -> Array:

        y = jnp.stack([x.max(axis=-1), x.mean(axis=-1)], axis=-1)
        y = nn.Conv(1, [self.filter_size, self.filter_size])(y)
        y = jax.nn.sigmoid(y)

        y = x * y

        return y


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    rate: float

    @nn.module.compact
    def __call__(
        self, inputs: ArrayLike, deterministic: Optional[bool] = True
    ) -> Array:
        if self.rate == 0.0:
            return inputs
        keep_prob = 1.0 - self.rate
        if deterministic:
            return inputs
        else:
            rng = self.make_rng("droppath")
            binary_factor = jnp.floor(
                keep_prob + jax.random.uniform(rng, dtype=inputs.dtype)
            )
            output = inputs / keep_prob * binary_factor
            return output


class FPN(nn.Module):
    out_channels: int = 256

    @nn.compact
    def __call__(self, inputs: Sequence[ArrayLike]) -> Sequence[Array]:
        out_channels = self.out_channels

        outputs = [jax.nn.relu(nn.Dense(out_channels)(x)) for x in inputs]

        for k in range(len(outputs) - 1, 0, -1):
            x = jax.image.resize(outputs[k], outputs[k - 1].shape, "nearest")
            x += outputs[k - 1]
            x = nn.Conv(out_channels, (3, 3))(x)
            x = jax.nn.relu(x)
            outputs[k - 1] = x

        return outputs
