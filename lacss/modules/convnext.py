from typing import Union,List,Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

from .common import *

''' Implements the convnext encoder. Described in https://arxiv.org/abs/2201.03545
Original implementation: https://github.com/facebookresearch/ConvNeXt
'''

class _Block(nn.Module):
    """ ConvNeXt Block.
    Args:
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    drop_rate: int = 0.
    layer_scale_init_value: float = 1e-6
    kernel_size:int = 7

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, training=None)->jnp.ndarray:
        dim = x.shape[-1]
        ks = self.kernel_size
        scale = self.layer_scale_init_value

        shortcut = x

        x = nn.Conv(dim, (ks,ks), feature_group_count=dim)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.Dense(dim * 4)(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(dim)(x)

        if scale > 0:
            gamma = self.param('gamma', lambda rng, shape: scale * jnp.ones(shape), (x.shape[-1]))
            x = x * gamma

        deterministic = training is None or not training
        x = DropPath(self.drop_rate)(x, deterministic=deterministic)

        x = x + shortcut

        return x

class ConvNeXt(nn.Module):
    """ ConvNeXt
    Args:
        patch_size: for stem default 4
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        out_channels (int): FPN output channels. Default: 256
    """

    patch_size:int = 4
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    drop_path_rate: float = 0.
    layer_scale_init_value: float = 1e-6
    out_channels: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, training:bool = None)->jnp.ndarray:
        dp_rate = 0
        outputs = []
        for k in range(len(self.depths)):
            if k == 0:
                ps = self.patch_size
                x = nn.Conv(self.dims[k], (ps, ps), strides=(ps, ps))(x)
                x = nn.LayerNorm(epsilon=1e-6)(x)
            else:
                x = nn.LayerNorm(epsilon=1e-6)(x)
                x = nn.Conv(self.dims[k], (2,2), strides=(2,2))(x)

            for _ in range(self.depths[k]):
                x = _Block(dp_rate, self.layer_scale_init_value)(x, training=training)
                dp_rate += self.drop_path_rate / (sum(self.depths) - 1)

            outputs.append(x)

        decoder_out = FPN(self.out_channels)(outputs)
        keys = [str(k + 1 if self.patch_size == 2 else k + 2) for k in range(4)]
        encoder_out = dict(zip(keys, outputs))
        decoder_out = dict(zip(keys, decoder_out))

        return encoder_out, decoder_out


# utility funcs

model_urls = {
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

# def load_weight(jax_model, url):
#     ''' Load pretrained convnext models
#     specs for pretrained models are:
#         tiny: dims=(96, 192, 384, 768), depths=(3,3,9,3)
#         small: dims=(96, 192, 384, 768), depths=(3,3,27,3)
#         base: dims=(128, 256, 512, 1024), depths=(3,3,27,3)
#         large: dims=(192, 384, 768, 1536), depths=(3,3,27,3)
#         X-large: dims=(256, 512, 1024, 2048), depths=(3,3,27,3)
#     '''
#     import torch

#     def t(m, k, new_value):
#         new_value = jnp.array(new_value)
#         assert vars(m)[k].shape == new_value.shape
#         vars(m)[k] = new_value

#     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#     params = checkpoint['model']

#     cnt = 1
#     for i in range(4):
#         for k in range(jax_model.depths[i]):
#             block = vars(jax_model)[f'block{cnt}' if cnt > 1 else 'block']
#             t(block, 'gamma', params[f'stages.{i}.{k}.gamma'])
#             t(block.conv, 'bias', params[f'stages.{i}.{k}.dwconv.bias'])
#             t(block.conv, 'kernel', params[f'stages.{i}.{k}.dwconv.weight'].permute(2,3,1,0))
#             t(block.layer_norm, 'bias', params[f'stages.{i}.{k}.norm.bias'])
#             t(block.layer_norm, 'scale', params[f'stages.{i}.{k}.norm.weight'])
#             t(block.linear, 'bias', params[f'stages.{i}.{k}.pwconv1.bias'])
#             t(block.linear, 'kernel', params[f'stages.{i}.{k}.pwconv1.weight'].transpose(0,1))
#             t(block.linear2, 'bias', params[f'stages.{i}.{k}.pwconv2.bias'])
#             t(block.linear2, 'kernel', params[f'stages.{i}.{k}.pwconv2.weight'].transpose(0,1))

#             cnt += 1

#     norm = jax_model.layer_norm
#     t(norm, 'bias', params['downsample_layers.0.1.bias'])
#     t(norm, 'scale', params['downsample_layers.0.1.weight'])

#     conv = jax_model.conv
#     t(conv, 'bias', params['downsample_layers.0.0.bias'])
#     t(conv, 'kernel', params['downsample_layers.0.0.weight'].permute(2,3,1,0))

#     for i in range(1,4):
#         norm = vars(jax_model)[f'layer_norm{i+1}']
#         t(norm, 'bias', params[f'downsample_layers.{i}.0.bias'])
#         t(norm, 'scale', params[f'downsample_layers.{i}.0.weight'])
#         conv = vars(jax_model)[f'conv{i+1}']
#         t(conv, 'bias', params[f'downsample_layers.{i}.1.bias'])
#         t(conv, 'kernel', params[f'downsample_layers.{i}.1.weight'].permute(2,3,1,0))

#     return jax_model
