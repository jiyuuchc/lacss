from typing import Union,List
import jax
import treex as tx
jnp = jax.numpy

from .types import *
from .se_net import ChannelAttention

''' Implements the convnext encoder. Described in https://arxiv.org/abs/2201.03545
Original implementation: https://github.com/facebookresearch/ConvNeXt
'''

class _Block(tx.Module):
    """ ConvNeXt Block.
    Args:
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    gamma: jnp.ndarray = tx.Parameter.node()
    key: Union[tx.Initializer, jnp.ndarray] = tx.Rng.node()
    
    def __init__(self, drop_rate=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        self.drop_rate = drop_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.kernel_size = kernel_size
        self.key = tx.Initializer(lambda key: jnp.array(key))

    def _drop_path(self, x):
        if self.training and self.drop_rate > 0:
            key, self.key = jax.random.split(self.key)
            keep_prob = 1.0 - self.drop_rate
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + jax.random.uniform(key, shape, dtype=x.dtype)
            random_tensor = jnp.floor(random_tensor)
            x = x / keep_prob * random_tensor
        return x

    @tx.compact
    def __call__(self, x: jnp.ndarray)->jnp.ndarray:
        dim = x.shape[-1]
        ks = self.kernel_size
        scale = self.layer_scale_init_value

        if self.initializing():
            self.gamma = scale * jnp.ones((dim))
            # self.key = tx.next_key()

        shortcut = x

        x = tx.Conv(dim, (ks,ks), feature_group_count=dim)(x)
        x = tx.LayerNorm(epsilon=1e-6)(x)
        x = tx.Linear(dim * 4)(x)
        x = jax.nn.gelu(x)
        x = tx.Linear(dim)(x)

        if scale > 0:
            x = x * self.gamma

        x = self._drop_path(x)

        x = x + shortcut

        return x

class _FPN(tx.Module):
    def __init__(
        self, 
        out_channels: int,
    ):
        super().__init__()
        self.out_channels = out_channels

    @tx.compact
    def __call__(self, inputs: List[jnp.ndarray]) -> List[jnp.ndarray] :
        out_channels = self.out_channels

        outputs = [jax.nn.relu(tx.Linear(out_channels)(x)) for x in inputs]

        for k in range(len(outputs)-1, 0, -1):
            x = jax.image.resize(outputs[k], outputs[k-1].shape, 'nearest')
            x += outputs[k-1]
            x = tx.Conv(out_channels, (3,3))(x)
            x = jax.nn.relu(x)
            outputs[k-1] = x

        return outputs

class ConvNeXt(tx.Module, ModuleConfig):
    """ ConvNeXt
    Args:
        patch_size: for stem default 4
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        out_channels (int): FPN output channels. Default: 256
    """
    def __init__(
        self,
        patch_size=4,
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0., 
        layer_scale_init_value=1e-6,
        out_channels=256,
    ):
        super().__init__()

        if patch_size != 2 and patch_size != 4:
            raise ValueError(f'patch_size must be 2 or 4, got {patch_size}')

        self._config_dict = dict(
            patch_size=patch_size,
            depths=depths, 
            dims=dims, 
            drop_path_rate=drop_path_rate, 
            layer_scale_init_value=layer_scale_init_value,
            out_channels=out_channels,
        )
    
    @tx.compact
    def __call__(self, x: jnp.ndarray)->jnp.ndarray:
        dp_rate = 0
        outputs = []
        for k in range(len(self.depths)):
            if k == 0:
                ps = self.patch_size
                x = tx.Conv(self.dims[k], (ps, ps), strides=(ps, ps))(x)
                x = tx.LayerNorm(epsilon=1e-6)(x)
            else:
                x = tx.LayerNorm(epsilon=1e-6)(x)
                x = tx.Conv(self.dims[k], (2,2), strides=(2,2))(x)

            for _ in range(self.depths[k]):
                x = _Block(dp_rate, self.layer_scale_init_value)(x)
                dp_rate += self.drop_path_rate / (sum(self.depths) - 1)

            outputs.append(x)

        decoder_out = _FPN(self.out_channels)(outputs)
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

def load_weight(jax_model, url):
    ''' Load pretrained convnext models
    specs for pretrained models are:
        tiny: dims=(96, 192, 384, 768), depths=(3,3,9,3)
        small: dims=(96, 192, 384, 768), depths=(3,3,27,3)
        base: dims=(128, 256, 512, 1024), depths=(3,3,27,3)
        large: dims=(192, 384, 768, 1536), depths=(3,3,27,3)
        X-large: dims=(256, 512, 1024, 2048), depths=(3,3,27,3)
    '''
    import torch

    def t(m, k, new_value):
        new_value = jnp.array(new_value)
        assert vars(m)[k].shape == new_value.shape
        vars(m)[k] = new_value

    checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
    params = checkpoint['model']

    cnt = 1
    for i in range(4):
        for k in range(jax_model.depths[i]):
            block = vars(jax_model)[f'block{cnt}' if cnt > 1 else 'block']
            t(block, 'gamma', params[f'stages.{i}.{k}.gamma'])
            t(block.conv, 'bias', params[f'stages.{i}.{k}.dwconv.bias'])
            t(block.conv, 'kernel', params[f'stages.{i}.{k}.dwconv.weight'].permute(2,3,1,0))
            t(block.layer_norm, 'bias', params[f'stages.{i}.{k}.norm.bias'])
            t(block.layer_norm, 'scale', params[f'stages.{i}.{k}.norm.weight'])
            t(block.linear, 'bias', params[f'stages.{i}.{k}.pwconv1.bias'])
            t(block.linear, 'kernel', params[f'stages.{i}.{k}.pwconv1.weight'].transpose(0,1))
            t(block.linear2, 'bias', params[f'stages.{i}.{k}.pwconv2.bias'])
            t(block.linear2, 'kernel', params[f'stages.{i}.{k}.pwconv2.weight'].transpose(0,1))

            cnt += 1

    norm = jax_model.layer_norm
    t(norm, 'bias', params['downsample_layers.0.1.bias'])
    t(norm, 'scale', params['downsample_layers.0.1.weight'])

    conv = jax_model.conv
    t(conv, 'bias', params['downsample_layers.0.0.bias'])
    t(conv, 'kernel', params['downsample_layers.0.0.weight'].permute(2,3,1,0))

    for i in range(1,4):
        norm = vars(jax_model)[f'layer_norm{i+1}']
        t(norm, 'bias', params[f'downsample_layers.{i}.0.bias'])
        t(norm, 'scale', params[f'downsample_layers.{i}.0.weight'])
        conv = vars(jax_model)[f'conv{i+1}']
        t(conv, 'bias', params[f'downsample_layers.{i}.1.bias'])
        t(conv, 'kernel', params[f'downsample_layers.{i}.1.weight'].permute(2,3,1,0))

    return jax_model
