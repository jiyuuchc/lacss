from typing import Sequence, Union, List, Tuple, Dict
import jax
import treex as tx
from .types import *

jnp = jax.numpy

from .se_net import ChannelAttention

class Bottleneck(tx.Module):
    key: Union[tx.Initializer, jnp.ndarray] = tx.Rng().node()

    def __init__(
        self,
        n_filters: int,
        strides: int = 1,
        dilation_rate: int = 1,
        se_ratio: int = 16,
        drop_rate: float = 0.0
    ) -> None:
        super().__init__()

        self.n_filters = n_filters
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.se_ratio = se_ratio
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
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
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        n_filters = self.n_filters
        strides = self.strides
        dilation = self.dilation_rate

        shortcut = inputs
        if strides > 1:
            shortcut = tx.Conv(n_filters * 4, (1,1), strides=(strides, strides), use_bias=False)(shortcut)
            shortcut= tx.GroupNorm(num_groups=None, group_size=1)(shortcut)

        x = inputs
        x = tx.Conv(n_filters, (1,1), use_bias=False)(x)
        x = tx.LayerNorm()(x)
        # x = tx.GroupNorm(num_groups=None, group_size=1)(x)
        x = jax.nn.relu(x)
        x = tx.Conv(n_filters, (3,3), strides=(strides, strides), input_dilation=(dilation, dilation), use_bias=False)(x)
        x = tx.LayerNorm()(x)
        # x = tx.GroupNorm(num_groups=None, group_size=1)(x)
        x = jax.nn.relu(x)
        x = tx.Conv(n_filters * 4, (1,1), use_bias=False)(x)
        x = tx.LayerNorm()(x)
        # x = tx.GroupNorm(num_groups=None, group_size=1)(x)
        x = jax.nn.relu(x)

        if self.se_ratio > 0:
            x = ChannelAttention(squeeze_factor=self.se_ratio)(x)

        x = self._drop_path(x)

        x = x + shortcut
        x = jax.nn.relu(x)

        return x

class FPN(tx.Module):
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

_RESNET_SPECS = {
  '50': [(64,3),(128,4),(256,6),(512,3)],
  '101': [(64,3),(128,4),(256,23),(512,3)],
  '152': [(64,3),(128,8),(256,36),(512,3)],
  '200': [(64,3),(128,24),(256,36),(512,3)],
  '270': [(64,4),(128,29),(256,53),(512,4)],
  '350': [(64,4),(128,36),(256,72),(512,4)],
  '420': [(64,4),(128,44),(256,87),(512,4)],
}

class ResNet(tx.Module, ModuleConfig):
    def __init__(
        self,
        model_spec: Union[str, List[Tuple[int, int]]] = '50',
        se_ratio: int = 16, 
        min_feature_level:int = 1, 
        out_channels:int = 256,
        stochastic_drop_rate = 0.0,
    ) -> None:

        super().__init__()

        self._config_dict = dict(
            model_spec = model_spec,
            se_ratio = se_ratio, 
            min_feature_level = min_feature_level,
            out_channels = out_channels,
            stochastic_drop_rate = stochastic_drop_rate,
        )
    
    @tx.compact
    def __call__(self, x: jnp.ndarray) -> dict:

        x = jax.nn.relu(tx.Conv(24, (3,3))(x))
        x = jax.nn.relu(tx.Conv(64, (3,3))(x))
        # x = tx.GroupNorm(num_groups=None, group_size=1)(x)
        encoder_out = [x]

        model_spec = self._config_dict['model_spec']
        if isinstance(model_spec, str):
            spec = _RESNET_SPECS[model_spec]
        else:
            spec = model_spec

        se_ratio = self._config_dict['se_ratio']
        stochastic_drop_rate = self._config_dict['stochastic_drop_rate'] if 'stochastic_drop_rate' in self._config_dict else 0.0
        for i, (n_filters, n_repeats) in enumerate(spec):
            drop_rate = stochastic_drop_rate * (i+2) / (len(spec) + 1)
            x = Bottleneck(n_filters, 2, se_ratio=se_ratio, drop_rate=drop_rate)(x)
            for _ in range(1, n_repeats):
                x = Bottleneck(n_filters, se_ratio=se_ratio, drop_rate=drop_rate)(x)
            encoder_out.append(x)

        l_min = self._config_dict['min_feature_level']
        out_channels = self._config_dict['out_channels']

        decoder_out = FPN(out_channels)(encoder_out[l_min:])

        keys = [str(k) for k in range(len(encoder_out))]
        encoder_out = dict(zip(keys, encoder_out))
        decoder_out = dict(zip(keys[l_min:], decoder_out))

        return encoder_out, decoder_out
