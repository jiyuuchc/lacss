from __future__ import annotations

import math
from functools import partial
from typing import Mapping, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from lacss.ops import gather_patches
from lacss.typing import *

from ..typing import *
from .common import SpatialAttention


class _Encoder(nn.Module):
    n_output_features: int
    input_patch_size: int
    encoding_dims: tuple[int, int, int] = (8, 8, 4)
    n_latent_features: int = -1

    @nn.compact
    def __call__(self, feature: ArrayLike, loc: ArrayLike) -> Array:
        patch_center, _, _, _ = gather_patches(feature, loc, patch_size=2)
        encodings = patch_center.mean(axis=(-2, -3))

        dim = self.n_latent_features
        dim = dim if dim > 0 else math.prod(self.encoding_dims)
        encodings = nn.Dense(dim)(encodings)
        encodings = jax.nn.relu(encodings)
        encodings = nn.Dense(dim)(encodings)
        encodings = jax.nn.relu(encodings)

        new_dim = math.prod(self.encoding_dims)
        encoding_shape = (-1,) + self.encoding_dims
        encodings = nn.Dense(new_dim)(encodings).reshape(encoding_shape)
        encodings = jax.image.resize(
            encodings,
            (
                encodings.shape[0],
                self.input_patch_size,
                self.input_patch_size,
                self.encoding_dims[-1],
            ),
            "linear",
        )

        encodings = nn.Conv(self.n_output_features, (1, 1), use_bias=False)(encodings)

        return encodings


class Segmentor(nn.Module):
    """LACSS segmentation head.

    Attributes:
        feature_level: The scale of the feature used for segmentation
        conv_spec: conv_block definition, e.g. ((384,384,384), (64,))
        instance_crop_size: Crop size for segmentation.
        with_attention: Whether use spatial attention layer.
        learned_encoding: Whether to use hard-coded position encoding.
        encoder_dims: Dim of the position encoder, if using learned encoding. Default is (8,8,4)

    """

    feature_level: int = 2
    conv_spec: tuple[Sequence[int], Sequence[int]] = (
        (384, 384, 384),
        (64,),
    )
    instance_crop_size: int = 96
    use_attention: bool = False
    learned_encoding: bool = True
    encoder_dims: Sequence[int] = (8, 8, 4)
    # n_cls: int = -1

    # if not feature_level in (0,1,2):
    #     raise ValueError('feature_level should be 1,2 or 0')

    @nn.compact
    def __call__(
        self, features: Mapping[str, ArrayLike], locations: ArrayLike
    ) -> DataDict:
        """
        Args:
            features: {'scale' [H, W, C]} feature dictionary from the backbone.
            locations: [N, 2]  normalized to image size.

        Returns:
            A dictionary of values representing segmentation outputs.

                * instance_output: [N, crop_size, crop_size]
                * instance_mask; [N, 1, 1] boolean mask indicating valid outputs
                * instance_yc: [N, crop_size, crop_size] meshgrid y coordinates
                * instance_xc: [N, crop_size, crop_size] meshgrid x coordinates
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
            encodings = _Encoder(
                n_ch,
                patch_size,
                self.encoder_dims,
            )(features[str(lvl)], locations)

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
