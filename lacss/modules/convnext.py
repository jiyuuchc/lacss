from __future__ import annotations

from functools import partial
from typing import Optional, Sequence, Callable

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import normalization
from flax.linen import activation

from ..typing import DataDict, Array, ArrayLike, Any
from .common import DropPath, DefaultUnpicklerMixin, ChannelAttention, GRN

""" Implements the convnext encoder. Described in https://arxiv.org/abs/2201.03545
Original implementation: https://github.com/facebookresearch/ConvNeXt
"""
_LayerNorm = partial(nn.LayerNorm, epsilon=1e-6)

class _Block(nn.Module):
    """ConvNeXt Block.
    Args:
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    drop_rate: float = 0.4
    kernel_size: int = 7
    se_ratio: int = 16
    normalization: Callable=normalization.GroupNorm
    activation: Callable[[ArrayLike], Array]=nn.gelu
    version: str = "1"
    dtype: Any = None

    @nn.compact
    def __call__(self, x: ArrayLike, *, deterministic: Optional[bool] = True) -> Array:
        dim = jnp.asarray(x).shape[-1]
        ks = self.kernel_size

        shortcut = x

        x = nn.Conv(dim, (ks, ks), feature_group_count=dim, dtype=self.dtype)(x)
        x = self.normalization(dtype=self.dtype)(x)
        x = nn.Dense(dim * 4, dtype=self.dtype)(x)
        x = self.activation(x)
        if self.version == "2":
            x = GRN()(x)
        x = nn.Dense(dim, dtype=self.dtype)(x)

        if self.se_ratio > 0:
            x = ChannelAttention(self.se_ratio, dtype=self.dtype)(x)

        x = DropPath(self.drop_rate)(x, deterministic=deterministic)

        x = x + shortcut

        return x


class ConvNeXt(nn.Module, DefaultUnpicklerMixin):
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
    kernel_size: int = 7
    normalization: Callable=nn.GroupNorm
    se_ratio: int = 0
    activation: Callable[[ArrayLike], Array]=nn.gelu
    deterministic: bool|None = None
    version: str = "1"
    dtype: Any = None

    @nn.compact
    def __call__(
        self, x: ArrayLike, *, deterministic: Optional[bool] = None,
    ) -> Sequence[Array]:
        """
        Args:
            x: Image input.
            training: Whether run the network in training mode (i.e. with stochastic depth)

        Returns:
            A list of feture arrays at different scaling, from higest to lowest resolution.
        """
        if deterministic is None:
            deterministic = self.deterministic
        dp_rate = 0
        outputs = []
        for k in range(len(self.depths)):
            if k == 0:
                ps = self.patch_size
                x = nn.Conv(self.dims[k], (ps, ps), strides=(ps, ps))(x)
                x = self.normalization()(x)
            else:
                x = self.normalization()(x)
                x = nn.Conv(self.dims[k], (2, 2), strides=(2, 2))(x)

            for _ in range(self.depths[k]):
                x = _Block(
                    dp_rate,
                    kernel_size=self.kernel_size,
                    se_ratio=self.se_ratio,
                    normalization=self.normalization,
                    activation=self.activation,
                    version=self.version,
                    dtype=self.dtype,
                )(x, deterministic=deterministic)
                dp_rate += self.drop_path_rate / (sum(self.depths) - 1)

            outputs.append(x)

        return outputs

    @classmethod
    def get_preconfigured(cls, model_type:str, patch_size=4, **kwargs):
        if model_type == "tiny":
            return cls(patch_size, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), se_ratio=16, **kwargs)
        elif model_type == "small":
            return cls(patch_size, depths=(3, 3, 27, 3), dims=(96, 192, 384, 768), se_ratio=16, **kwargs)
        elif model_type == "base":
            return cls(patch_size, depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024), se_ratio=16, **kwargs)
        elif model_type == "large":
            return cls(patch_size, depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536), se_ratio=16, **kwargs)
        elif model_type == "x-large":
            return cls(patch_size, depths=(3, 3, 27, 3), dims=(256, 512, 1024, 2048), se_ratio=16, **kwargs)
        elif model_type == "tiny_v2":
            return cls(
                patch_size, 
                depths=(3, 3, 9, 3), 
                dims=(96, 192, 384, 768),
                se_ratio=-1,
                normalization=_LayerNorm,
                version="2",
                **kwargs,
            )
        elif model_type == "base_v2":
            return cls(
                patch_size, 
                depths=(3, 3, 27, 3), 
                dims=(128, 256, 512, 1024),
                se_ratio=-1,
                normalization=_LayerNorm,
                version="2",
                **kwargs,
            )
        elif model_type == "large_v2":
            return cls(
                patch_size, 
                depths=(3, 3, 27, 3), 
                dims=(192, 384, 768, 1536),
                se_ratio=-1,
                normalization=_LayerNorm,
                version="2",
                **kwargs,
            )
        elif model_type == "huge_v2":
            return cls(
                patch_size, 
                depths=(3, 3, 27, 3), 
                dims=(352, 704, 1408, 2816),
                se_ratio=-1,
                normalization=_LayerNorm,
                version="2",
                **kwargs,
            )            
        else:
            raise ValueError(f"unknown model type {model_type}")
