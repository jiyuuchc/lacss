from typing import Union, Tuple
import jax
import treex as tx
jnp = jax.numpy

from .types import *

_unet_configs = {
    'unet_s': (32, 64, 128, 256, 512),
    'unet_n': (64, 128, 256, 512, 1024),
    'unet_l': (96, 192, 384, 768, 1536),
}

class MixingBlock(tx.Module):
    def __init__(
        self, 
        method='upsampling', 
        hidden_dim=None, 
    ):
        super().__init__()
        self.method = method
        self.hidden_dim = hidden_dim

    @tx.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        method = self.method
        hidden_dim = self.hidden_dim

        if method == 'conv_add':
            hidden_dim = y.shape[-1]
        if method == 'conv_concat' and hidden_dim is None:
            hidden_dim = x.shape[-1] // 2

        if method == 'conv_add' or method == 'conv_concat':
            x = tx.ConvTranspose(hidden_dim, (3,3), strides=(2,2))(x)

        elif self.method == 'upsampling':
            x = jax.image.resize(x, y.shape[:3] + x.shape[-1:], 'linear')

        else:
            raise ValueError('Valid methods are conv_add|conv_concat|upsampling')

        if method == 'conv_add':
            x = x + y
        else:
            x = jnp.concatenate((x,y), axis=-1)

        return x

class UNet(tx.Module, ModuleConfig):
    def __init__(
        self, 
        net_spec: Union[str, Tuple[int]] = 'unet_n', 
        method: str = 'conv_add', 
        min_feature_level: int = 1
    ):
        super().__init__()

        self._config_dict = dict(
            net_spec = net_spec, 
            method = method, 
            min_feature_level = min_feature_level,
        )
    
    def maxpool(self, x):
        b,h,w,c = x.shape
        x = x.reshape(b,h//2, 2, w, c).max(2)
        x = x.reshape(b,h//2, w//2, 2, c).max(3)
        return x

    @tx.compact
    def __call__(self, x: jnp.ndarray) -> dict:
        output = []

        if isinstance(self.net_spec, str):
            net_spec = _unet_configs[net_spec]

        for k, n_ch in enumerate(net_spec):
            for _ in range(2):
                x = tx.Conv(n_ch, (3,3), use_bias=True)(x)
                x = jax.nn.relu(x)
            output.append(x)
    
            if k < len(net_spec)-1:
                x = self.maxpool(x)

        l_min = self.min_feature_level

        for k in range(len(output)-1, l_min, -1):
            x = MixingBlock(self.method)(output[k], output[k-1])
            n_ch = net_spec[k-1]
            for _ in range(2):
                x = tx.Conv(n_ch, (3,3), use_bias=True)(x)
                x = jax.nn.relu(x)
            output[k-1] = x

        keys = [str(k) for k in range(l_min, len(output)+l_min)]
        output = dict(zip(keys, output[l_min:]))

        return output
