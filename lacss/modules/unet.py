from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..typing import Array, ArrayLike

class UNet(nn.Module):
    model_spec: Sequence[int] = (32, 64, 128, 256, 512)
    patch_size: int = 1
    se_ratio: int = -1
    is_stack: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.patch_size in [1, 2, 4]:
            raise ValueError("patch_size must be eith 1, 2 or 4")

    def _stack_conv(self, x):
        if self.is_stack:
            x = nn.Conv(x.shape[-1], (3, 1, 1), use_bias=False)(x)
        return x

    @nn.compact
    def __call__(self, x: ArrayLike) -> Sequence[Array]:

        encoder_out = []
        fs = max(self.patch_size, 3)
        st = self.patch_size

        ra = tuple(range(x.ndim))

        for ch in self.model_spec:

            x = self._stack_conv(nn.Conv(ch, (fs, fs), (st, st), use_bias=False)(x))
            x = nn.GroupNorm(num_groups=ch, use_scale=False)(x[None, ...])[0]
            x = jax.nn.relu(x)

            x = self._stack_conv(nn.Conv(ch, (3, 3), use_bias=False)(x))
            x = nn.GroupNorm(num_groups=ch, use_scale=False)(x[None, ...])[0]
            x = jax.nn.relu(x)

            encoder_out.append(x)

            fs, st = 3, 2

        decoder_out = [x]

        for y in encoder_out[-2::-1]:

            ch = y.shape[-1]

            x = jax.image.resize(x, y.shape[:-1] + x.shape[-1:], "linear")
            x = jnp.concatenate([x, y], axis=-1)

            x = self._stack_conv(nn.Conv(ch, (3, 3), use_bias=False)(x))
            x = nn.GroupNorm(num_groups=ch, use_scale=False)(x[None, ...])[0]
            x = jax.nn.relu(x)

            x = self._stack_conv(nn.Conv(ch, (3, 3), use_bias=False)(x))
            x = nn.GroupNorm(num_groups=ch, use_scale=False)(x[None, ...])[0]
            x = jax.nn.relu(x)

            decoder_out.insert(0, x)

        return decoder_out
