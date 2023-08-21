from __future__ import annotations

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..typing import DataDict, Params
from .common import *

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

    @nn.compact
    def __call__(self, x: ArrayLike, *, training: Optional[bool] = None) -> Array:
        dim = x.shape[-1]
        ks = self.kernel_size
        scale = self.layer_scale_init_value

        shortcut = x

        x = nn.Conv(dim, (ks, ks), feature_group_count=dim)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.Dense(dim * 4)(x)
        x = jax.nn.gelu(x)
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
    depths: Sequence[int] = (3, 3, 27, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 1e-6
    out_channels: int = 384

    @nn.compact
    def __call__(
        self, x: ArrayLike, *, training: Optional[bool] = None
    ) -> tuple[DataDict, DataDict]:
        """
        Args:
            x: Image input.
            training: Whether run the network in training mode (i.e. with stochastic depth)

        Returns:
            A tuple of (encoder_outputs, decoder_outputs). Both are dictionaries mapping
                feature scale (e.g. "2") to features. If out_channels is -1, the decoder_output
                is None.
        """
        dp_rate = 0
        outputs = []
        for k in range(len(self.depths)):
            if k == 0:
                ps = self.patch_size
                x = nn.Conv(self.dims[k], (ps, ps), strides=(ps, ps))(x)
                x = nn.LayerNorm(epsilon=1e-6)(x)
            else:
                x = nn.LayerNorm(epsilon=1e-6)(x)
                x = nn.Conv(self.dims[k], (2, 2), strides=(2, 2))(x)

            for _ in range(self.depths[k]):
                x = _Block(dp_rate, self.layer_scale_init_value)(x, training=training)
                dp_rate += self.drop_path_rate / (sum(self.depths) - 1)

            outputs.append(x)

        keys = [str(k + 1 if self.patch_size == 2 else k + 2) for k in range(4)]
        encoder_out = dict(zip(keys, outputs))

        if self.out_channels > 0:
            decoder_out = FPN(self.out_channels)(outputs)
            decoder_out = dict(zip(keys, decoder_out))
        else:
            decoder_out = None

        return encoder_out, decoder_out

    def _load_weight(self, jax_params, url):
        import torch
        from flax.core.frozen_dict import freeze, unfreeze

        def t(m, new_value):
            new_value = jnp.array(new_value)
            assert m.shape == new_value.shape
            m = new_value

        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        params = checkpoint["model"]
        jax_params = unfreeze(jax_params)

        cnt = 0
        for i in range(4):
            for k in range(self.depths[i]):
                block = jax_params[f"_Block_{cnt}"]
                t(block["gamma"], params[f"stages.{i}.{k}.gamma"])
                t(block["Conv_0"]["bias"], params[f"stages.{i}.{k}.dwconv.bias"])
                t(
                    block["Conv_0"]["kernel"],
                    params[f"stages.{i}.{k}.dwconv.weight"].permute(2, 3, 1, 0),
                )
                t(block["LayerNorm_0"]["bias"], params[f"stages.{i}.{k}.norm.bias"])
                t(block["LayerNorm_0"]["scale"], params[f"stages.{i}.{k}.norm.weight"])
                t(block["Dense_0"]["bias"], params[f"stages.{i}.{k}.pwconv1.bias"])
                t(
                    block["Dense_0"]["kernel"],
                    params[f"stages.{i}.{k}.pwconv1.weight"].transpose(0, 1),
                )
                t(block["Dense_1"]["bias"], params[f"stages.{i}.{k}.pwconv2.bias"])
                t(
                    block["Dense_1"]["kernel"],
                    params[f"stages.{i}.{k}.pwconv2.weight"].transpose(0, 1),
                )

                cnt += 1

        norm = jax_params[f"LayerNorm_0"]
        t(norm["bias"], params["downsample_layers.0.1.bias"])
        t(norm["scale"], params["downsample_layers.0.1.weight"])

        conv = jax_params[f"Conv_0"]
        t(conv["bias"], params["downsample_layers.0.0.bias"])
        t(conv["kernel"], params["downsample_layers.0.0.weight"].permute(2, 3, 1, 0))

        for i in range(1, 4):
            norm = jax_params[f"LayerNorm_{i}"]
            t(norm["bias"], params[f"downsample_layers.{i}.0.bias"])
            t(norm["scale"], params[f"downsample_layers.{i}.0.weight"])
            conv = jax_params[f"Conv_{i}"]
            t(conv["bias"], params[f"downsample_layers.{i}.1.bias"])
            t(
                conv["kernel"],
                params[f"downsample_layers.{i}.1.weight"].permute(2, 3, 1, 0),
            )

        return freeze(jax_params)

    def get_imagenet_weights(self, model_type: str) -> Params:
        """Get imagenet weights

        Args:
            model_type: The expected model specification. This must match the current
                instance attributes.

                * tiny: dims=(96, 192, 384, 768), depths=(3,3,9,3)
                * small: dims=(96, 192, 384, 768), depths=(3,3,27,3)
                * base: dims=(128, 256, 512, 1024), depths=(3,3,27,3)
                * large: dims=(192, 384, 768, 1536), depths=(3,3,27,3)
                * X-large: dims=(256, 512, 1024, 2048), depths=(3,3,27,3)

        Returns:
            A frozen dict representing weights of the current module
        """

        init_params = self.init(jax.random.PRNGKey(0), jnp.zeros([64, 64, 3]))
        init_params = init_params["params"]
        params = self._load_weight(
            init_params,
            _imagenet_weights_urls[f"convnext_{model_type}_22k"],
        )

        return params
