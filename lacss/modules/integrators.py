from __future__ import annotations

from functools import partial
from typing import Callable, Sequence, Any

import flax.linen as nn

import jax

from .common import DefaultUnpicklerMixin, ChannelAttention, picklable_relu
from ..typing import Array, ArrayLike


class FPN(nn.Module, DefaultUnpicklerMixin):
    out_channels: int = 384
    se_ratio: int = 0
    activation: Callable[[Array], Array] = picklable_relu
    dtype: Any = None

    @nn.compact
    def __call__(self, inputs: Sequence[ArrayLike]) -> Sequence[Array]:
        out_channels = self.out_channels

        outputs = [self.activation(nn.Dense(out_channels, dtype=self.dtype)(x)) for x in inputs]

        if self.se_ratio > 0:
            outputs = [ChannelAttention(self.se_ratio, dtype=self.dtype)(x) for x in outputs]

        for k in range(len(outputs) - 1, 0, -1):
            x = jax.image.resize(outputs[k], outputs[k - 1].shape, "nearest")
            x += outputs[k - 1]
            x = nn.Conv(out_channels, (3, 3), dtype=self.dtype)(x)
            x = self.activation(x)
            outputs[k - 1] = x

        return outputs


class ConvIntegrator(nn.Module, DefaultUnpicklerMixin):
    """A UNet-like feature integrator.

    Unlike FPN, its feature spaces are not consistent at different spatial scales.
    high-res scale should have lower number of features (like unet)
    """

    n_features: Sequence[int] | None = None
    n_layers: int = 1
    norm: Callable = nn.GroupNorm
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: Sequence[Array]) -> Sequence[Array]:
        n_features = self.n_features or [f.shape[-1] for f in x]

        y = x[-1]
        for _ in range(self.n_layers):
            y = nn.Conv(n_features[-1], (3, 3))(y)
            y = self.activation(y)
        output = [y]

        for k in range(-1, -1 * len(x), -1):
            y = nn.ConvTranspose(
                x[k - 1].shape[-1],
                (3, 3),
                strides=(2, 2),
            )(output[0])

            y = self.norm()(x[k - 1] + y)

            for _ in range(self.n_layers):
                y = nn.Conv(n_features[k - 1], (3, 3))(y)
                y = self.activation(y)

            output.insert(0, y)

        return output
