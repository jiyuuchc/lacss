import math
from typing import List
import jax
import treex as tx
from .unet import MixingBlock
from .types import *

jnp = jax.numpy

''' XCiT model
    based on https://github.com/facebookresearch/xcit
    ref: https://arxiv.org/abs/2106.09681
'''

class PositionalEncodingFourier(tx.Module):
    ''' positonal encoding as the original tansformer model
    '''
    def __init__(self, hidden_dim=32, temperature=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    @tx.compact
    def __call__(self, x):
        height, width, out_dims = x.shape[-3:]
        hidden_dim = self.hidden_dim

        dim_t = self.temperature ** (jnp.arange(hidden_dim//2) / (hidden_dim//2))
        mesh = jnp.mgrid[:height, :width].transpose(1,2,0)  #[height, width, 2]
        mesh = mesh / jnp.array([height, width]) * 2 * math.pi + 1e-6
        mesh = jnp.expand_dims(mesh, -1) / dim_t
        mesh = jnp.concatenate((jnp.sin(mesh), jnp.cos(mesh)), axis=-1) # [height, width, 2, hidden_dim//2]

        mesh = mesh.reshape(height, width, hidden_dim * 2)

        pos = tx.Linear(out_dims)(mesh)
        x = x + pos

        return x

class ConvPatchEmbed(tx.Module):
    ''' Image to patch embedding
    '''
    def __init__(self, patch_size=8, embed_dim=384):
        super().__init__()

        if patch_size != 16 and patch_size != 8 and patch_size != 4:
            raise ValueError('patch_size must be 4|8|16')

        self.patch_size = patch_size
        self.embed_dim = embed_dim

    @tx.compact
    def __call__(self, x:jnp.ndarray) -> List[jnp.ndarray]:
        patch_size = self.patch_size
        n_ch = self.embed_dim * 2 // patch_size

        x = tx.Conv(n_ch, (3,3), strides=(2,2), use_bias=False)(x)
        x = tx.GroupNorm(num_groups=n_ch)(x)
        x = jax.nn.gelu(x)
        outputs = [x]  
        x = tx.Conv(n_ch * 2, (3,3), strides=(2,2), use_bias=False)(x)
        x = tx.GroupNorm(num_groups=n_ch * 2)(x)
        x = jax.nn.gelu(x)
        outputs.append(x) 

        if patch_size >= 8:
            x = tx.Conv(n_ch * 4, (3,3), strides=(2,2), use_bias=False)(x)
            x = tx.GroupNorm(num_groups=n_ch * 4)(x)
            x = jax.nn.gelu(x)
            outputs.append(x)

        if patch_size >= 16:
            x = tx.Conv(n_ch * 8, (3,3), strides=(2,2), use_bias=False)(x)
            x = tx.GroupNorm(num_groups=n_ch * 8)(x)
            x = jax.nn.gelu(x)
            outputs.append(x)

        return outputs

class LPI(tx.Module):
    ''' Local patch interaction layer
    Use depth-wide conv to mixing feature from adjacient patches
    '''

    def __init__(self, kernel_size = 3):
        super().__init__()
        self.kernel_size = kernel_size

    @tx.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        n_ch = x.shape[-1]
        ks = self.kernel_size

        # original implementation use conv-act-norm
        # here it is conv-norm-act
        x = tx.Conv(n_ch, (ks, ks), feature_group_count=n_ch, use_bias=False)(x)  
        x = tx.GroupNorm(num_groups=n_ch)(x)
        x = jax.nn.gelu(x)

        x = tx.Conv(n_ch, (ks, ks), feature_group_count=n_ch)(x)

        return x

class MLP(tx.Module):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters

    @tx.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_ch = x.shape[-1]
        for n_ch in self.filters:
            x = tx.Linear(n_ch)(x)
            x = jax.nn.gelu(x)
        x = tx.Linear(in_ch)(x)

        return x

class XCA(tx.Module):
    ''' Cross-Covariance Attention
    '''

    temp: jnp.ndarray = tx.Parameter.node()

    def __init__(self, num_heads=8, use_bias=False):
        super().__init__()

        self.use_bias = use_bias
        self.num_heads = num_heads

        self.temp = jnp.ones([num_heads, 1, 1], float)

    @tx.compact
    def __call__(self, x: jnp.ndarray)->jnp.ndarray:
        height, width, in_dim = x.shape
        use_bias = self.use_bias
        num_heads = self.num_heads
        n_ch = in_dim // num_heads

        x = x.reshape(height * width, in_dim)
        x = tx.Linear(in_dim * 3, use_bias=use_bias)(x).reshape(-1, num_heads, n_ch, 3)
        q, k, v = x[..., 0], x[..., 1], x[..., 2]

        q = jax.nn.normalize(q, axis=0) # norm along all locations 
        k = jax.nn.normalize(k, axis=0)
        attn = (q.transpose(1,2,0) @ k.transpose(1,0,2)) * self.temp
        attn = jax.nn.softmax(attn)

        x = v.transpose(1,0,2) @ attn  #[heads, patches, n_ch]
        x = x.transpose(1,0,2).reshape(height, width, in_dim)
        x = tx.Linear(in_dim)(x)

        return x

class XCABlock(tx.Module):
    def __init__(self, num_heads=8, use_bias=False, mlp_filters=1536):
        super().__init__()

        self.num_heads = num_heads
        self.use_bias = use_bias  
        self.mlp_filters = mlp_filters

    @tx.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # layer_norm apply only to feature dim as in the orig code
        x = tx.LayerNorm()(x)
        x = jax.vmap(XCA(self.num_heads, self.use_bias))(x)
        x = tx.LayerNorm()(x)
        x = LPI()(x)
        x = tx.LayerNorm()(x)
        x = MLP([self.mlp_filters])(x)

        return x

class XCiT(tx.Module, ModuleConfig):
    def __init__(self, min_feature_level=1, patch_size=8, embed_dim=384, depth=12, n_heads=8, use_bias=True, mlp_ratio=4):
        '''
        Args:
            min_feature_level: 
            patch_size:
            embed_dim:
            depth:
            n_heads:
            use_bias:
            mlp_ratio:
        '''
        super().__init__()
        self._config_dict = dict(
            min_feature_level=min_feature_level, 
            patch_size=patch_size, 
            embed_dim=embed_dim, 
            depth=depth, 
            n_heads=n_heads, 
            use_bias=use_bias,
            mlp_ratio=mlp_ratio,
        )
    
    @tx.compact
    def __call__(self, x: jnp.ndarray) -> list:
        config = self.get_config()

        outputs = [x] + ConvPatchEmbed(config['patch_size'], config['embed_dim'])(x)

        x = outputs[-1]
        x = PositionalEncodingFourier()(x)

        mlp_filters = int(config['embed_dim'] * config['mlp_ratio'])
        for i in range(config['depth']):
            x = XCABlock(config['n_heads'], config['use_bias'], mlp_filters)(x)
        outputs[-1] = x

        # decode by mixing levels
        min_lvl = config['min_feature_level']
        for k in range(len(outputs)-1, min_lvl, -1):
            n_ch = outputs[k-1].shape[-1]
            x = MixingBlock('conv_add')(outputs[k], outputs[k-1])
            x = tx.Conv(n_ch, (3,3))(x)
            x = jax.nn.gelu(x)
            outputs[k-1] = x

        keys = [str(k) for k in range(min_lvl, len(outputs))]
        decoder_out = dict(zip(keys, outputs[min_lvl:]))

        return decoder_out
