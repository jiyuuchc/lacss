from typing import Tuple
from functools import partial
import jax
import treex as tx

from .se_net import SpatialAttention
from ..ops import gather_patches
from .types import *

jnp = jax.numpy

class _Encoder(tx.Module):
    def __init__(self, n_ch, patch_size, resolution=8, encoding_ch=4):
        super().__init__()
        self.n_ch = n_ch
        self.patch_size = patch_size
        self.resolution = resolution
        self.encoding_ch = 4
    
    @tx.compact
    def __call__(self, feature, loc):

        patch_center, _, _, _ = gather_patches(feature, loc, patch_size=2)
        encodings = patch_center.mean(axis=(-2,-3))

        dim = self.encoding_ch * self.resolution * self.resolution
        # encodings = tx.Dropout(0.5)(encodings)
        encodings = tx.Linear(dim)(encodings)
        encodings = jax.nn.relu(encodings)
        encodings = tx.Linear(dim)(encodings)
        encodings = jax.nn.relu(encodings)

        encoding_shape = (-1, self.resolution, self.resolution, self.encoding_ch)
        encodings = tx.Linear(dim)(encodings).reshape(encoding_shape)
        encodings = jax.image.resize(encodings, [encodings.shape[0], self.patch_size, self.patch_size, self.encoding_ch], 'linear')

        encodings = tx.Conv(self.n_ch, (1,1), use_bias=False)(encodings)
        
        return encodings

class Segmentor(tx.Module, ModuleConfig):
    # mix_bias: jnp.ndarray = tx.Parameter.node()
    # ra_avg: jnp.ndarray = tx.BatchStat.node()
    # ra_var: jnp.ndarray = tx.BatchStat.node()

    def __init__(
        self,
        feature_level: int = 1,
        conv_spec: Tuple[Tuple[int], Tuple[int]] = ((64, 64, 64), (16,),),
        instance_crop_size: int = 96,
        use_attention: bool = False,
        learned_encoding: bool = False,
        # masked_batchnorm: bool = True,
    ):
        """
          Args:
            conv_spec: conv_block definition, e.g. ((64, 64, 64), (16, 16))
            instance_crop_size: crop size, int
            feature_scale_ratio: 1,2 or 4
            with_attention: T/F whether use spatial attention layer
            learned_encoding: Use hard-coded position encoding or not
        """
        super().__init__()

        if not feature_level in (0,1,2):
            raise ValueError('feature_level should be 1,2 or 0')

        self._config_dict = dict(
            feature_level = feature_level,
            conv_spec = conv_spec,
            instance_crop_size = instance_crop_size,
            use_attention = use_attention,
            learned_encoding = learned_encoding,
        )
    
    @tx.compact
    def __call__(self, features: dict, locations: jnp.ndarray) -> tuple:
        '''
        Args:
            features: {'lvl' [B, H, W, C]} feature dictionary
            locations: [B, N, 2]  scaled 0..1
        outputs:
            instance_output: [B, N, crop_size, crop_size, 1] 
            yc: [B, N, crop_size, crop_size]
            xc: [B, N, crop_size, crop_size]
        '''
        crop_size = self._config_dict['instance_crop_size']
        lvl = self._config_dict['feature_level']
        scale = 2 ** lvl
        conv_spec = self._config_dict['conv_spec']
        patch_size = crop_size // scale

        x = features[str(lvl)]
        _, h, w, _ = x.shape

        #feature convs
        for n_ch in conv_spec[0]:
            x = tx.Conv(n_ch, (3,3), use_bias=False)(x)
            x = tx.GroupNorm(num_groups=n_ch)(x)
            # x = tx.BatchNorm()(x)
            x = jax.nn.relu(x)

        # mixing features with pos_encodings
        n_ch = conv_spec[1][0]
        # x = tx.Conv(n_ch, (1,1), use_bias=False)(x)
        x = tx.Conv(n_ch, (3,3), use_bias=False)(x)
        gather_op = partial(gather_patches, patch_size=patch_size)
        patches, y0, x0, _ = jax.vmap(gather_op)(x, locations)

        if not self._config_dict['learned_encoding']:
            encodings = (jnp.mgrid[:patch_size, :patch_size] / patch_size - .5).transpose(1,2,0)
            encodings = tx.Conv(n_ch, (1,1), use_bias=False)(encodings)
        else:
            encodings = jax.vmap(_Encoder(n_ch, patch_size))(features[str(lvl)], locations)
        patches += encodings
        mask = jnp.expand_dims((locations >= 0).any(axis=-1), (2,3))

        # norm
        # if self.training:
        #     avg = jnp.mean(patches, where=mask, axis=(0,1,2,3))
        #     var = jnp.var(patches, where=mask, axis=(0,1,2,3))
        #     self.ra_avg = 0.99 * self.ra_avg + 0.01 * avg
        #     self.ra_var = 0.99 * self.ra_var + 0.01 * var
        # else:
        #     avg = self.ra_avg
        #     var = self.ra_var
        # patches = (patches - avg) * jax.lax.rsqrt(var + 1e-5) + self.mix_bias
        # patch_shape = patches.shape
        # patches = patches.reshape(patch_shape[:-3] + (-1,))
        # patches = tx.LayerNorm()(patches)
        # patches = patches.reshape(patch_shape)
        patches = jax.nn.relu(patches)

        if self._config_dict['use_attention']:
            patches = SpatialAttention()(patches)

        # patche convs
        for n_ch in conv_spec[1][1:]:
            patches = jax.vmap(tx.Conv(n_ch, (3,3)))(patches)
            patches = jax.nn.relu(patches)

        # outputs
        if scale == 1:
            logits = jax.vmap(tx.Conv(1, (3,3)))(patches)
        else:
            logits = jax.vmap(tx.ConvTranspose(1, (2,2), strides=(2,2)))(patches)
            if scale == 4:
                logits = jax.image.resize(logits, patches.shape[:2] + (crop_size, crop_size, 1), 'linear')

        logits = logits.squeeze(-1)
        outputs = jax.nn.sigmoid(logits)

        # indicies
        yc, xc = jnp.mgrid[:crop_size, :crop_size]
        yc = yc + y0[:, :, None, None] * scale
        xc = xc + x0[:, :, None, None] * scale

        # clear invalid locations
        outputs = jnp.where(mask, outputs, 0)
        logits = jnp.where(mask, logits, 0)
        xc = jnp.where(mask, xc, -1)
        yc = jnp.where(mask, yc, -1)

        return dict(
            instance_output = outputs,
            instance_yc = yc,
            instance_xc = xc,
            instance_logtis = logits,
            instance_mask = mask,
        )
