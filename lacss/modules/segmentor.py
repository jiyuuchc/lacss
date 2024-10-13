from __future__ import annotations

import math
from typing import Sequence, Callable, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import normalization

from ..typing import ArrayLike, Array
from .common import DefaultUnpicklerMixin, ChannelAttention

class Segmentor(nn.Module, DefaultUnpicklerMixin):
    """LACSS segmentation head.

    Attributes:
        n_cls: num of classes
        feature_scale: the spatail scale of the feature level
        instance_crop_size: Crop size for segmentation.
        pos_emb_shape: Dim of the learned position encoder.
    """
    feature_scale: int = 4
    instance_crop_size: int = 96
    patch_dim: int = 32
    sig_dim: int = 512
    pos_emb_shape: Sequence[int] = (16, 16, 4)
    dtype: Any = None

    @property
    def patch_size(self):
        return self.instance_crop_size // self.feature_scale

    def _get_patch(self, feature, locations)->tuple[Array, Array]:
        feature = nn.Conv(self.patch_dim, (3,3), dtype=self.dtype)(feature)

        patch_shape = (self.patch_size, self.patch_size)
        locations = locations.astype(int) - jnp.asarray(patch_shape) // 2  # so we can use loc for indexing
        coords = jnp.mgrid[:patch_shape[0], :patch_shape[1]] + locations[..., None, None] # N, 2, PS, PS

        limit = jnp.expand_dims(jnp.asarray(feature.shape[:-1]), (1,2))

        valid_locs = (coords >= 0).all(axis=1) & (coords < limit).all(axis=1) # N, PS, PS
        patches = jnp.where(
            valid_locs[..., None],
            feature[coords[:, 0], coords[:, 1]],
            0,
        )
        return patches, locations


    def _get_signiture(self, patches):
        bw = self.patch_size // 4
        n, _, _, n_ch = patches.shape
        x = jax.image.resize(
            patches[:, bw:-bw, bw:-bw, :],
            (n, 8, 8, n_ch),
            "linear",
        ).reshape(n, -1)

        x = nn.LayerNorm(dtype=self.dtype)(nn.Dense(self.sig_dim, dtype=self.dtype)(x))

        return x


    def _add_pos_encoding(self, patches, sig_vec):
        x = sig_vec
        x = nn.gelu(nn.Dense(self.sig_dim, dtype=self.dtype)(x))
        x = nn.gelu(nn.Dense(self.sig_dim, dtype=self.dtype)(x))

        x = nn.Dense(math.prod(self.pos_emb_shape), dtype=self.dtype)(x)
        x = jax.image.resize(
            x.reshape(-1, *self.pos_emb_shape),
            (patches.shape[0], self.patch_size, self.patch_size, self.pos_emb_shape[-1]),
            "linear",
        )

        encoding = nn.Dense(patches.shape[-1], use_bias=False, dtype=self.dtype)(x) #[n, ps, ps, dim]

        patches = jax.nn.relu(patches + encoding)

        return patches


    @nn.compact
    def __call__(self, feature: ArrayLike, locations: ArrayLike) -> dict:
        """
        Args:
            features: [H, W, C] features from the backbone.
            locations: [N, 2]

        Returns:
            A nested dictionary of values representing segmentation outputs.
              * segmentor/logits: all segment logits
              * predictions/segmentations: logits of designated class [N, D, ps, ps]
              * predictions/segmentation_y0_coord: y coord of patch top-left corner [N]
              * predictions/segmentation_x0_coord: x coord of patch top-left corner [N]
              * predictions/segmentation_is_valid: mask for valid patches [N]
        """
        locations = jnp.asarray(locations) / self.feature_scale

        x = jnp.asarray(feature)

        patches, patch_locs = self._get_patch(x, locations) # N, PS, PS, ch

        patches = ChannelAttention(dtype=self.dtype)(patches)

        patch_sigs = self._get_signiture(patches)

        patches = self._add_pos_encoding(patches, patch_sigs)

        logits = nn.ConvTranspose(1, (3, 3), strides=(2, 2), dtype=self.dtype)(patches)

        output_shape = (
            locations.shape[0],
            self.instance_crop_size,
            self.instance_crop_size
        )
        logits = jax.image.resize(logits.squeeze(-1), output_shape, "linear")
        instance_mask = locations[:, -1] >= 0
        patch_locs = jnp.where(
            instance_mask[:, None],
            patch_locs * self.feature_scale,
            -1,
        )

        return dict(
            segmentor=dict(
                patch_signitures=patch_sigs,
            ),
            predictions=dict(
                segmentations=logits[:, None, :, :], #[N, 1, ps, ps]
                segmentation_y0_coord=patch_locs[:, 0],
                segmentation_x0_coord=patch_locs[:, 1],
                segmentation_z0_coord=jnp.zeros_like(patch_locs[:, 0]),
                segmentation_is_valid=instance_mask,
            ),
        )
