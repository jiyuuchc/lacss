from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..typing import *
from .common import *


class UNet(nn.Module):
    model_spec: Sequence[int] = (32, 64, 128, 256, 512)
    patch_size: int = 1
    se_ratio: int = -1

    def __post_init__(self):
        super().__post_init__()
        if not self.patch_size in [1, 2, 4]:
            raise ValueError("patch_size must be eith 1, 2 or 4")

    @nn.compact
    def __call__(self, x: ArrayLike) -> tuple[DataDict, DataDict]:

        encoder_out = []
        fs = max(self.patch_size, 3)
        st = self.patch_size

        ra = tuple(range(x.ndim))

        for ch in self.model_spec:

            x = nn.Conv(ch, (fs, fs), (st, st), use_bias=False)(x)
            # x = nn.LayerNorm(epsilon=1e-6)(x)
            x = nn.GroupNorm(num_groups=ch, use_scale=False)(x[None, ...])[0]
            x = jax.nn.relu(x)

            x = nn.Conv(ch, (3, 3), use_bias=False)(x)
            # x = nn.LayerNorm(epsilon=1e-6)(x)
            x = nn.GroupNorm(num_groups=ch, use_scale=False)(x[None, ...])[0]
            x = jax.nn.relu(x)

            encoder_out.append(x)

            fs, st = 3, 2

        decoder_out = [x]

        for y in encoder_out[-2::-1]:

            ch = y.shape[-1]

            x = jax.image.resize(x, y.shape[:-1] + x.shape[-1:], "linear")
            x = jnp.concatenate([x, y], axis=-1)

            x = nn.Conv(ch, (3, 3), use_bias=False)(x)
            # x = nn.LayerNorm(epsilon=1e-6)(x)
            x = nn.GroupNorm(num_groups=ch, use_scale=False)(x[None, ...])[0]
            x = jax.nn.relu(x)

            x = nn.Conv(ch, (3, 3), use_bias=False)(x)
            # x = nn.LayerNorm(epsilon=1e-6)(x)
            x = nn.GroupNorm(num_groups=ch, use_scale=False)(x[None, ...])[0]
            x = jax.nn.relu(x)

            decoder_out.insert(0, x)

        keys = [str(lvl + self.start_level) for lvl in range(len(self.model_spec))]

        encoder_out = dict(zip(keys, encoder_out))
        decoder_out = dict(zip(keys, decoder_out))

        return encoder_out, decoder_out

    @property
    def start_level(self) -> int:
        if self.patch_size == 4:
            return 2
        elif self.patch_size == 2:
            return 1
        else:
            return 0
