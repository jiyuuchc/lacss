from functools import partial
from typing import Dict, List, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..ops import gather_patches
from .common import SpatialAttention


class _Encoder(nn.Module):
    n_ch: int
    patch_size: int
    resolution: int = 8
    encoding_ch: int = 4

    @nn.compact
    def __call__(self, feature, loc):

        patch_center, _, _, _ = gather_patches(feature, loc, patch_size=2)
        encodings = patch_center.mean(axis=(-2, -3))

        dim = self.encoding_ch * self.resolution * self.resolution
        encodings = nn.Dense(dim)(encodings)
        encodings = jax.nn.relu(encodings)
        encodings = nn.Dense(dim)(encodings)
        encodings = jax.nn.relu(encodings)

        encoding_shape = (-1, self.resolution, self.resolution, self.encoding_ch)
        encodings = nn.Dense(dim)(encodings).reshape(encoding_shape)
        encodings = jax.image.resize(
            encodings,
            [encodings.shape[0], self.patch_size, self.patch_size, self.encoding_ch],
            "linear",
        )

        encodings = nn.Conv(self.n_ch, (1, 1), use_bias=False)(encodings)

        return encodings


class Segmentor(nn.Module):
    """
    Args:
    conv_spec: conv_block definition, e.g. ((64, 64, 64), (16, 16))
    instance_crop_size: crop size, int
    feature_scale_ratio: 1,2 or 4
    with_attention: T/F whether use spatial attention layer
    learned_encoding: Use hard-coded position encoding or not
    """

    feature_level: int = 1
    conv_spec: Tuple[Sequence[int], Sequence[int]] = (
        (64, 64, 64),
        (16,),
    )
    instance_crop_size: int = 96
    use_attention: bool = False
    learned_encoding: bool = False
    n_cls: int = -1

    # if not feature_level in (0,1,2):
    #     raise ValueError('feature_level should be 1,2 or 0')

    @nn.compact
    def __call__(self, features: dict, locations: jnp.ndarray) -> tuple:
        """
        Args:
            features: {'lvl' [H, W, C]} feature dictionary
            locations: [N, 2]  scaled 0..1
        outputs:
            instance_output: [N, crop_size, crop_size, 1]
            yc: [N, crop_size, crop_size]
            xc: [N, crop_size, crop_size]
        """
        crop_size = self.instance_crop_size
        lvl = self.feature_level
        scale = 2**lvl
        conv_spec = self.conv_spec
        patch_size = crop_size // scale

        x = features[str(lvl)]
        # h, w, _ = x.shape

        # feature convs
        for n_ch in conv_spec[0]:
            x = nn.Conv(n_ch, (3, 3), use_bias=False)(x)
            x = nn.GroupNorm(num_groups=n_ch)(x[None, ...])[0]
            x = jax.nn.relu(x)

        # mixing features with pos_encodings
        n_ch = conv_spec[1][0]
        x = nn.Conv(n_ch, (3, 3), use_bias=False)(x)
        patches, y0, x0, _ = gather_patches(x, locations, patch_size=patch_size)

        if not self.learned_encoding:
            encodings = (
                jnp.mgrid[:patch_size, :patch_size] / patch_size - 0.5
            ).transpose(1, 2, 0)
            encodings = nn.Conv(n_ch, (1, 1), use_bias=False)(encodings)
        else:
            encodings = _Encoder(n_ch, patch_size)(features[str(lvl)], locations)
        patches += encodings
        mask = jnp.expand_dims((locations >= 0).any(axis=-1), (1, 2))
        patches = jax.nn.relu(patches)

        if self.use_attention:
            patches = SpatialAttention()(patches)

        # patche convs
        for n_ch in conv_spec[1][1:]:
            patches = nn.Conv(n_ch, (3, 3))(patches)
            patches = jax.nn.relu(patches)

        # outputs
        if scale == 1:
            logits = nn.Conv(1, (3, 3))(patches)
        else:
            logits = nn.ConvTranspose(1, (2, 2), strides=(2, 2))(patches)
            if scale == 4:
                logits = jax.image.resize(
                    logits, patches.shape[:1] + (crop_size, crop_size, 1), "linear"
                )

        logits = logits.squeeze(-1)
        outputs = jax.nn.sigmoid(logits)

        # indicies
        yc, xc = jnp.mgrid[:crop_size, :crop_size]
        yc = yc + y0[:, None, None] * scale
        xc = xc + x0[:, None, None] * scale

        # clear invalid locations
        outputs = jnp.where(mask, outputs, 0)
        logits = jnp.where(mask, logits, 0)
        xc = jnp.where(mask, xc, -1)
        yc = jnp.where(mask, yc, -1)

        return dict(
            instance_output=outputs,
            instance_yc=yc,
            instance_xc=xc,
            instance_logit=logits,
            instance_mask=mask,
        )
