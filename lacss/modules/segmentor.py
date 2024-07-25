from __future__ import annotations

import math
from typing import Sequence, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import normalization
from flax.linen import activation

from lacss.ops import sub_pixel_samples

from ..typing import ArrayLike, Array
from .common import SpatialAttention, picklable_relu


class Segmentor(nn.Module):
    """LACSS segmentation head.

    Attributes:
        feature_level: the feature level used for segmentation
        feature_scale: the spatail scale of the feature level
        conv_spec: conv_block definition, e.g. ((384,384,384), (64,))
        instance_crop_size: Crop size for segmentation.
        with_attention: Whether use spatial attention layer.
        encoder_dims: Dim of the learned position encoder.
    """
    feature_level: int = 0
    feature_scale: int = 4
    conv_spec: tuple[Sequence[int], Sequence[int]] = (
        (192, 192, 192),
        (48,),
    )
    normalization: Callable[[None], nn.Module]=normalization.GroupNorm
    activation: Callable[[Array], Array] = picklable_relu

    n_cls: int = 1
    instance_crop_size: int = 96
    use_attention: bool = False
    encoder_dims: Sequence[int] = (8, 8, 4)
    max_embedding_length:int = 16

    @property
    def patch_size(self):
        return self.instance_crop_size // self.feature_scale

    def _gather_patches(self, feature, locs, patch_size=None):
        patch_size = patch_size or self.patch_size
        locs = locs.astype(int) # so we can use loc for indexing

        yy, xx = (
            jnp.mgrid[:patch_size, :patch_size] 
            - patch_size // 2
            + locs[:, :, None, None]
        ).swapaxes(0, 1) # [2, N, PS, PS]

        # padding to avoid out-of-bound
        padding_size = patch_size // 2 + 1
        paddings = [
            [padding_size, padding_size],
            [padding_size, padding_size],
            [0, 0],
        ]
        padded_feature = jnp.pad(feature, paddings)

        patches = padded_feature[yy + padding_size, xx + padding_size, :]

        return patches, yy, xx

    def _pos_encoder(self, feature, locations):
        # derive pos embedding (YX) from feature values at cell location
        # the rationale is that the feature value already has some info regarding
        # cell size/shape
        patch_size = self.patch_size
        depth = feature.shape[0]

        sampling_locs = jnp.floor(locations) - jnp.asarray([0, 0.5, 0.5]) # center of the patch
        x = sub_pixel_samples(feature, sampling_locs)
        assert x.shape == (locations.shape[0], feature.shape[-1])

        for _ in range(2):
            x = nn.Dense(x.shape[-1])(x)
            x = nn.relu(x)

        dim_out = math.prod(self.encoder_dims[-3:])
        xa = nn.Dense(dim_out)(x)
        xa = xa.reshape((-1,) + self.encoder_dims[-3:])

        xa = jax.image.resize(
            xa,
            (
                x.shape[0],
                patch_size,
                patch_size,
                self.encoder_dims[-1],
            ),
            "linear",
        )

        encodings = nn.Dense(self.conv_spec[1][0], use_bias=False)(xa)

        if len(self.encoder_dims) == 3:
            z_encodings = self._z_encodings(depth, locations[:, 0]) # [D, N, c]

        else:
            z_dim = self.encoder_dims[0] * self.encoder_dims[-1]
            xb = nn.Dense(z_dim)(x)
            xb = xb.reshape(-1, self.encoder_dims[0], self.encoder_dims[-1])
            xb = jax.image.resize(
                xb,
                (
                    x.shape[0],
                    self.max_embedding_length,
                    self.encoder_dims[-1],
                ),
                "linear",
            )
            embeddings = nn.Dense(self.conv_spec[1][0], use_bias=False)(xb) # [N, L, c]

            z_idx = self.max_embedding_length // 2 - locations[:, 0].astype(int) # [N]
            z_idx = jnp.clip(
                z_idx[:, None] + jnp.arange(depth), 
                0, self.max_embedding_length -1
            ) # [N, D]

            z_encodings = jax.vmap(lambda b, z: b[z])(embeddings, z_idx) # [N, D, c]
            z_encodings = z_encodings.swapaxes(0, 1) # [D, N, c]
                        
        encodings = encodings + z_encodings[:, :, None, None, :] #[D, N, ps, ps, c]

        return encodings #[D, N, ps, ps, c]

    def _z_encodings(self, depth, z_locs):
        # z_pos_embedding is a simple learned matrix
        max_embedding_length = self.max_embedding_length
        target_dim = self.conv_spec[1][0]

        pos_embedding = self.param(
            "z_embedding",
            nn.initializers.normal(stddev=0.02),
            (max_embedding_length, target_dim),
        )

        z_idx = max_embedding_length // 2 - z_locs.astype(int)
        z_idx = jnp.clip(
            z_idx[:, None] + jnp.arange(depth),
            0, self.max_embedding_length -1,
        ) # [N, D]

        pos_embedding = jax.vmap(lambda z: pos_embedding[z])(z_idx) # [N, D, n_ch]
        pos_embedding = pos_embedding.swapaxes(0, 1) # [D, N, n_ch]

        return pos_embedding #[D, N, n_ch]

    @nn.compact
    def __call__(
        self,
        features: list[ArrayLike],
        locations: ArrayLike,
        classes: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        *,
        training=False,
    ) -> dict:
        """
        Args:
            features: list[array: [D, H, W, C]] multiscale features from the backbone.
            locations: [N, 3]
            classes: [N] optional

        Returns:
            A nested dictionary of values representing segmentation outputs.
              * segmentor/logits: all segment logits
              * predictions/segmentations: logits of designated class [N, D, ps, ps]
              * predictions/segmentation_y0_coord: y coord of patch top-left corner [N]
              * predictions/segmentation_x0_coord: x coord of patch top-left corner [N]
              * predictions/segmentation_is_valid: mask for valid patches [N]
        """
        assert features[self.feature_level].ndim == 4
        assert locations.ndim == 2
        assert locations.shape[-1] == 3

        crop_size = self.instance_crop_size
        locations = locations / jnp.asarray([1, self.feature_scale, self.feature_scale])

        # feature convs
        x = features[self.feature_level]
        D, H, W, C = x.shape

        for n_ch in self.conv_spec[0]:
            x = nn.Conv(n_ch, (3, 3))(x)
            x = self.normalization()(x)
            x = self.activation(x)

        x = nn.Conv(self.conv_spec[1][0], (3, 3))(x)

        self.sow("intermediates", "segmentor_image_features", x) # [D, H, W, C]

        # pos encoder
        feature = features[self.feature_level]
        pos_encodings = self._pos_encoder(feature, locations) # [D, N, ps, ps, c]

        self.sow("intermediates", "segmentor_pos_embed", pos_encodings)

        # mixing features with pos_encodings
        patches, ys, xs = jax.vmap(
            self._gather_patches,
            in_axes=(0, None),
            out_axes=(0, None, None),
        )(x, locations[:, 1:]) # [D, N, ps, ps, c]

        patches = jax.nn.relu(patches + pos_encodings) 

        if self.use_attention:
            patches = SpatialAttention()(patches)

        # patche convs
        for n_ch in self.conv_spec[1][1:]:
            patches = nn.Conv(n_ch, (3, 3))(patches)
            patches = self.activation(patches)

        self.sow("intermediates", "segmentor_patch_features", patches)

        # outputs
        if self.feature_scale == 1:
            logits = nn.Conv(self.n_cls, (3, 3))(patches)
        else:
            logits = nn.ConvTranspose(self.n_cls, (2, 2), strides=(2, 2))(patches)
            if self.feature_scale != 2:
                logits = jax.image.resize(
                    logits,
                    patches.shape[:2] + (crop_size, crop_size, self.n_cls),
                    "linear",
                )

        outputs = jnp.take_along_axis(
            logits,
            classes.reshape(1, -1, 1, 1, 1),
            axis=-1,
        ) # [D, N, ps, ps, 1]
        outputs = outputs.swapaxes(0, 1).squeeze(-1) #[N, D, ps, ps]

        # clear invalid locations, needed for the semi-supervied loss functions
        instance_mask = (locations >= 0).all(axis=-1)
        y0 = jnp.where(instance_mask, ys[:, 0, 0] * self.feature_scale, -1)
        x0 = jnp.where(instance_mask, xs[:, 0, 0] * self.feature_scale, -1)
        yc, xc = jnp.mgrid[:self.instance_crop_size, :self.instance_crop_size]
        yc = yc + y0[:, None, None] # [N, ps, ps]
        xc = xc + x0[:, None, None]
        valid_pixels = (xc >= 0) & (yc >=0) & (xc < W * self.feature_scale) & (yc < H * self.feature_scale)
        if mask is not None:
            valid_pixels &= mask.any(axis=0)[yc, xc] # [N, ps, ps]
            valid_pixels = jnp.expand_dims(valid_pixels, 1) & mask.any(axis=(1,2), keepdims=True) #[N, D, ps, ps]
        else:
            valid_pixels = jnp.expand_dims(valid_pixels, 1)
        valid_pixels &= jnp.expand_dims(instance_mask, (1,2,3))
        outputs = jnp.where(
            valid_pixels,
            outputs,
            -1e8,
        )

        return dict(
            segmentor=dict(
                logits=logits,
            ),
            predictions=dict(
                segmentations=outputs, #[N, D, ps, ps]
                segmentation_y0_coord=y0,
                segmentation_x0_coord=x0,
                segmentation_is_valid=instance_mask,
            ),
        )
