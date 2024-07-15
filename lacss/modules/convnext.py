from __future__ import annotations

from typing import Optional, Sequence, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..typing import DataDict, Array, ArrayLike
from .common import DropPath

""" Implements the convnext encoder. Described in https://arxiv.org/abs/2201.03545
Original implementation: https://github.com/facebookresearch/ConvNeXt
"""


class _Block(nn.Module):
    """ConvNeXt Block.
    Args:
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    drop_rate: int = 0.4
    layer_scale_init_value: float = 1e-6
    kernel_size: int = 7
    normalization: Callable[[None], nn.Module]=nn.LayerNorm
    activation: Callable[[Array], Array]=nn.gelu

    @nn.compact
    def __call__(self, x: ArrayLike, *, training: Optional[bool] = None) -> Array:
        dim = x.shape[-1]
        ks = self.kernel_size
        scale = self.layer_scale_init_value

        shortcut = x

        x = nn.Conv(dim, (ks, ks), feature_group_count=dim)(x)
        x = self.normalization()(x)
        x = nn.Dense(dim * 4)(x)
        x = self.activation(x)
        x = nn.Dense(dim)(x)

        if scale > 0:
            gamma = self.param(
                "gamma", lambda rng, shape: scale * jnp.ones(shape), (x.shape[-1])
            )
            x = x * gamma

        deterministic = training is None or not training
        x = DropPath(self.drop_rate)(x, deterministic=deterministic)

        x = x + shortcut

        return x


# utility funcs

_imagenet_weights_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


class ConvNeXt(nn.Module):
    """ConvNeXt CNN backbone

    Attributes:
        patch_size: Stem patch size
        depths: Number of blocks at each stage.
        dims: Feature dimension at each stage.
        drop_path_rate: Stochastic depth rate.
        layer_scale_init_value: Init value for Layer Scale.
        out_channels:
            FPN output channels. Setting this to -1 disable the FPN, in which case
            the model output only encoder outputs.
    """

    patch_size: int = 4
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    drop_path_rate: float = 0.4
    layer_scale_init_value: float = 1e-6
    kernel_size: int = 7
    normalization: Callable[[None], nn.Module]=nn.LayerNorm
    activation: Callable[[Array], Array]=nn.gelu

    @nn.compact
    def __call__(
        self, x: ArrayLike, *, training: Optional[bool] = None
    ) -> tuple[DataDict, DataDict]:
        """
        Args:
            x: Image input.
            training: Whether run the network in training mode (i.e. with stochastic depth)

        Returns:
            A list of feture arrays at different scaling, from higest to lowest resolution.
        """
        dp_rate = 0
        outputs = []
        for k in range(len(self.depths)):
            if k == 0:
                ps = self.patch_size
                x = nn.Conv(self.dims[k], (ps, ps), strides=(ps, ps))(x)
                # FIXME normalization?
                # x = nn.LayerNorm(epsilon=1e-6)(x)
            else:
                x = self.normalization()(x)
                x = nn.Conv(self.dims[k], (2, 2), strides=(2, 2))(x)

            for _ in range(self.depths[k]):
                x = _Block(
                    dp_rate,
                    layer_scale_init_value=self.layer_scale_init_value,
                    kernel_size=self.kernel_size,
                    normalization=self.normalization,
                    activation=self.activation,
                )(x, training=training)
                dp_rate += self.drop_path_rate / (sum(self.depths) - 1)

            outputs.append(x)

        return outputs

    @classmethod
    def get_model_tiny(cls, patch_size=4):
        return cls(patch_size, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))

    @classmethod
    def get_model_small(cls, patch_size=4):
        return cls(patch_size, depths=(3, 3, 27, 3), dims=(96, 192, 384, 768))

    @classmethod
    def get_model_base(cls, patch_size=4):
        return cls(patch_size, depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024))

    @classmethod
    def get_model_large(cls, patch_size=4):
        return cls(patch_size, depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536))

    @classmethod
    def get_model_x_large(cls, patch_size=4):
        return cls(patch_size, depths=(3, 3, 27, 3), dims=(256, 512, 1024, 2048))
