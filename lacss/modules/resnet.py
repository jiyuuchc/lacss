from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..typing import *
from .common import *


class Bottleneck(nn.Module):
    n_filters: int
    strides: int = 1
    se_ratio: int = 16
    drop_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, training: bool = None) -> jnp.ndarray:
        n_filters = self.n_filters
        strides = self.strides

        shortcut = inputs
        if strides > 1:
            shortcut = nn.Conv(
                n_filters * 4, (1, 1), strides=(strides, strides), use_bias=False
            )(shortcut)
            shortcut = nn.GroupNorm(num_groups=None, group_size=1)(shortcut)

        x = inputs
        x = nn.Conv(n_filters, (1, 1), use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = jax.nn.relu(x)
        x = nn.Conv(n_filters, (3, 3), strides=(strides, strides), use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = jax.nn.relu(x)
        x = nn.Conv(n_filters * 4, (1, 1), use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = jax.nn.relu(x)

        if self.se_ratio > 0:
            x = ChannelAttention(squeeze_factor=self.se_ratio)(x)

        deterministic = training is None or not training
        x = DropPath(self.drop_rate)(x, deterministic=deterministic)

        x = x + shortcut
        x = jax.nn.relu(x)

        return x


_RESNET_SPECS = {
    "50": [(64, 3), (128, 4), (256, 6), (512, 3)],
    "101": [(64, 3), (128, 4), (256, 23), (512, 3)],
    "152": [(64, 3), (128, 8), (256, 36), (512, 3)],
    "200": [(64, 3), (128, 24), (256, 36), (512, 3)],
    "270": [(64, 4), (128, 29), (256, 53), (512, 4)],
    "350": [(64, 4), (128, 36), (256, 72), (512, 4)],
    "420": [(64, 4), (128, 44), (256, 87), (512, 4)],
}


class ResNet(nn.Module):
    model_spec: Union[str, Sequence[Tuple[int, int]]] = "50"
    se_ratio: int = 16
    min_feature_level: int = 1
    stochastic_drop_rate = 0.0
    out_channels: int = 256
    patch_size: int = 2

    def __post_init__(self):

        if self.patch_size != 2 and self.patch_size != 4:
            raise ValueError(
                "Invalud patch_size {self.patch_size}. It shoule be either 2 or 4."
            )

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, training: bool = None) -> dict:

        if self.patch_size == 2:
            x = jax.nn.relu(nn.Conv(24, (3, 3))(x))
            x = jax.nn.relu(nn.Conv(64, (3, 3))(x))
            encoder_out = [x]
        else:
            encoder_out = [x]
            x = jax.nn.relu(nn.Conv(24, (3, 3), strides=(2, 2))(x))
            x = jax.nn.relu(nn.Conv(64, (3, 3))(x))
            encoder_out.append(x)

        model_spec = self.model_spec
        if isinstance(model_spec, str):
            spec = _RESNET_SPECS[model_spec]
        else:
            spec = model_spec

        se_ratio = self.se_ratio
        stochastic_drop_rate = self.stochastic_drop_rate
        for i, (n_filters, n_repeats) in enumerate(spec):
            drop_rate = stochastic_drop_rate * (i + 2) / (len(spec) + 1)
            x = Bottleneck(n_filters, 2, se_ratio=se_ratio, drop_rate=drop_rate)(
                x, training=training
            )
            for _ in range(1, n_repeats):
                x = Bottleneck(n_filters, se_ratio=se_ratio, drop_rate=drop_rate)(
                    x, training=training
                )
            encoder_out.append(x)

        # l_min = self.min_feature_level
        # out_channels = self.out_channels

        # decoder_out = FPN(out_channels)(encoder_out[l_min:])

        # keys = [str(k) for k in range(len(encoder_out))]
        # encoder_out = dict(zip(keys, encoder_out))
        # decoder_out = dict(zip(keys[l_min:], decoder_out))

        return encoder_out
