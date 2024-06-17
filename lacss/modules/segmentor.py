from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from lacss.ops import sub_pixel_samples

from ..typing import ArrayLike
from .common import SpatialAttention


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

    feature_level: int = 0
    feature_scale: int = 4
    conv_spec: tuple[Sequence[int], Sequence[int]] = (
        (384, 384, 384),
        (64,),
    )
    n_cls: int = 1
    instance_crop_size: int = 96
    use_attention: bool = False
    learned_encoding: bool = True
    encoder_dims: Sequence[int] = (8, 8, 4)

    @property
    def patch_size(self):
        return self.instance_crop_size // self.feature_scale

    def _gather_patches(self, feature, locs, patch_size=None):
        patch_size = patch_size or self.patch_size

        yy, xx = jnp.mgrid[:patch_size, :patch_size] - patch_size // 2
        xx = xx + locs[:, 1, None, None]
        yy = yy + locs[:, 0, None, None]

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

    def _static_pos_encoder(self):
        patch_size = self.patch_size

        pos_encodings = (
            jnp.mgrid[:patch_size, :patch_size] / patch_size - 0.5
        ).transpose(1, 2, 0)

        return pos_encodings

    def _pos_encoder(self, feature, locations):
        patch_size = self.patch_size

        encodings, _, _ = self._gather_patches(feature, locations, 2)
        encodings = encodings.mean(axis=(1, 2))

        dim = math.prod(self.encoder_dims)
        encodings = nn.Dense(dim)(encodings)
        encodings = jax.nn.relu(encodings)
        encodings = nn.Dense(dim)(encodings)
        encodings = jax.nn.relu(encodings)
        encodings = nn.Dense(dim)(encodings)

        encoding_shape = (-1,) + self.encoder_dims
        encodings = encodings.reshape(encoding_shape)

        # reshape to match patch size
        encodings = jax.image.resize(
            encodings,
            (
                encodings.shape[0],
                patch_size,
                patch_size,
                self.encoder_dims[-1],
            ),
            "linear",
        )

        return encodings

    @nn.compact
    def __call__(
        self,
        features: list[ArrayLike],
        locations: ArrayLike,
        classes: ArrayLike | None = None,
    ) -> dict:
        """
        Args:
            features: list[array: [H, W, C]] multiscale features from the backbone.
            locations: [N, 2]
            classes: [N] optional

        Returns:
            A dictionary of values representing segmentation outputs.
                * instance_output: [N, crop_size, crop_size]
                * instance_mask; [N, 1, 1] boolean mask indicating valid outputs
                * instance_yc: [N, crop_size, crop_size] meshgrid y coordinates
                * instance_xc: [N, crop_size, crop_size] meshgrid x coordinates
        """
        crop_size = self.instance_crop_size
        locations = (locations / self.feature_scale).astype(int)

        x = features[self.feature_level]

        # pos encoder
        if not self.learned_encoding:
            pos_encodings = self._static_pos_encoder()
        else:
            pos_encodings = self._pos_encoder(x, locations)

        pos_encodings = nn.Conv(self.conv_spec[1][0], (1, 1), use_bias=False)(
            pos_encodings
        )

        # feature convs
        for n_ch in self.conv_spec[0]:
            x = nn.Conv(n_ch, (3, 3), use_bias=False)(x)
            x = nn.LayerNorm(use_scale=False)(x)
            # x = nn.GroupNorm(num_groups=n_ch)(x[None, ...])[0]
            x = jax.nn.relu(x)

        self.sow("intermediates", "segmentor_image_features", x)
        self.sow("intermediates", "segmentor_pos_embed", pos_encodings)

        # mixing features with pos_encodings
        x = nn.Conv(self.conv_spec[1][0], (3, 3), use_bias=False)(x)
        patches, ys, xs = self._gather_patches(x, locations)
        patches = jax.nn.relu(patches + pos_encodings)

        if self.use_attention:
            patches = SpatialAttention()(patches)

        # patche convs
        for n_ch in self.conv_spec[1][1:]:
            patches = nn.Conv(n_ch, (3, 3))(patches)
            patches = jax.nn.relu(patches)

        self.sow("intermediates", "segmentor_patch_features", patches)

        # outputs
        if self.feature_scale == 1:
            logits = nn.Conv(self.n_cls, (3, 3))(patches)
        else:
            logits = nn.ConvTranspose(self.n_cls, (2, 2), strides=(2, 2))(patches)
            if self.feature_scale != 2:
                logits = jax.image.resize(
                    logits,
                    patches.shape[:1] + (crop_size, crop_size, self.n_cls),
                    "linear",
                )

        if classes is None:
            outputs = logits[..., :1]
        else:
            outputs = jnp.take_along_axis(
                logits,
                classes.reshape(-1, 1, 1, 1),
                axis=-1,
            )

        # indicies
        yc, xc = jnp.mgrid[: self.instance_crop_size, : self.instance_crop_size]
        yc = yc + ys[:, :1, :1] * self.feature_scale
        xc = xc + xs[:, :1, :1] * self.feature_scale

        # clear invalid locations
        mask = (locations >= 0).all(axis=-1)
        mask = jnp.expand_dims(mask, (1, 2))

        return dict(
            segmentor=dict(
                logits=logits,
            ),
            predictions=dict(
                segmentations=outputs.squeeze(-1),
                segmentation_y_coords=yc,
                segmentation_x_coords=xc,
                segmentation_is_valid=mask,
            ),
        )
