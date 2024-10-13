from __future__ import annotations

import math
from typing import Sequence, Callable, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import normalization

from ..typing import ArrayLike, Array
from .common import DefaultUnpicklerMixin, ChannelAttention

class Segmentor3D(nn.Module, DefaultUnpicklerMixin):
    """LACSS 3D segmentation head.

    Attributes:
        n_layers: num of conv layers for feature mixing
        feature_scale: the spatail scale of the feature level
        instance_crop_size: Crop size for segmentation.
        pos_emb_shape: Dim of the learned position encoder.
    """
    n_layers: int = 1

    feature_scale: int = 4
    instance_crop_size: int = 96
    patch_dim: int = 48
    sig_dim: int = 1024
    pos_emb_shape: Sequence[int] = (8, 8, 8, 4)
    dtype: Any = None

    @property
    def patch_size(self):
        return self.instance_crop_size // self.feature_scale

    def _get_patch(self, feature, locations):
        patch_shape = (self.patch_size,) * 3
        locations = locations.astype(int) - jnp.asarray(patch_shape) // 2  # so we can use loc for indexing
        coords = jnp.mgrid[:self.patch_size, :self.patch_size, :self.patch_size] + locations[..., None, None, None] # N, 3, PS, PS, PS

        limit = jnp.expand_dims(jnp.asarray(feature.shape[:-1]), (1,2,3))

        valid_locs = (coords >= 0).all(axis=1) & (coords < limit).all(axis=1) # N, PS, PS
        patches = jnp.where(
            valid_locs[..., None],
            feature[coords[:, 0], coords[:, 1], coords[:, 2]],
            0,
        )
        return patches, locations

    def _get_signiture(self, patches):
        bw = self.patch_size // 4
        n, _, _, _, n_ch = patches.shape
        x = jax.image.resize(
            patches[:, bw:-bw, bw:-bw, bw:-bw, :],
            (n, 4, 4, 4, n_ch),
            "linear",
        ).reshape(n, -1)

        x = nn.Dense(self.sig_dim, dtype=self.dtype)(x)
        x = nn.LayerNorm(dtype=self.dtype)(x)

        return x


    def _add_pos_encoding(self, patches, sig_vec):
        x = sig_vec
        x = nn.gelu(nn.Dense(self.sig_dim, dtype=self.dtype)(x))
        x = nn.gelu(nn.Dense(self.sig_dim, dtype=self.dtype)(x))

        x = nn.Dense(math.prod(self.pos_emb_shape), dtype=self.dtype)(x)
        x = jax.image.resize(
            x.reshape(-1, *self.pos_emb_shape),
            (patches.shape[0], self.patch_size, self.patch_size, self.patch_size, self.pos_emb_shape[-1]),
            "linear",
        )

        encoding = nn.Dense(patches.shape[-1], use_bias=False, dtype=self.dtype)(x) #[n, ps, ps, ps, dim]

        # xa_dim = self.pos_emb_res * self.pos_emb_res * self.pos_emb_dim
        # xa = nn.Dense(xa_dim, dtype=self.dtype)(x)
        # xa = jax.image.resize(
        #     xa.reshape(-1, self.pos_emb_res, self.pos_emb_res, self.pos_emb_dim),
        #     (xa.shape[0], self.patch_size, self.patch_size, self.pos_emb_dim),
        #     "linear",
        # )
        # xa = nn.Dense(patches.shape[-1], use_bias=False, dtype=self.dtype)(xa) #[n, ps, ps, dim]

        # xb_dim = self.pos_emb_res * self.pos_emb_dim
        # xb = nn.Dense(xb_dim, dtype=self.dtype)(x)
        # xb = jax.image.resize(
        #     xb.reshape(-1, self.pos_emb_res, self.pos_emb_dim),
        #     (xa.shape[0], self.patch_size, self.pos_emb_dim),
        #     "linear",
        # )
        # xb = nn.Dense(patches.shape[-1], use_bias=False, dtype=self.dtype)(xb) #[n, ps, dim]

        # encoding = xa[:, None, :, :, :] + xb[:, :, None, None, :]

        patches = jax.nn.relu(patches + encoding)

        return patches


    @nn.compact
    def __call__(self, feature: ArrayLike, locations: ArrayLike) -> dict:
        """
        Args:
            features: [D, H, W, C] image feature
            locations: [N, 3]

        Returns:
            A nested dictionary of values representing segmentation outputs.
              * segmentor/logits: all segment logits
              * predictions/segmentations: logits of designated class [N, D, ps, ps]
              * predictions/segmentation_y0_coord: y coord of patch top-left corner [N]
              * predictions/segmentation_x0_coord: x coord of patch top-left corner [N]
              * predictions/segmentation_is_valid: mask for valid patches [N]
        """
        locations = locations / self.feature_scale

        x = feature
        dim = self.patch_dim
        for _ in range(self.n_layers):
            x = nn.Conv(dim, (3,3,3), dtype=self.dtype)(x)
            x = nn.gelu(x)
        x = nn.Conv(dim, (3,3,3), dtype=self.dtype)(x)

        patches, patch_locs = self._get_patch(x, locations) # N, PS, PS, PS, ch

        patches = ChannelAttention(dtype=self.dtype)(patches)

        patch_sigs = self._get_signiture(patches)

        patches = self._add_pos_encoding(patches, patch_sigs)

        logits = nn.ConvTranspose(1, (2, 2, 2), strides=(2, 2, 2), dtype=self.dtype)(patches)

        output_shape = (locations.shape[0],) + (self.instance_crop_size,) * 3
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
                segmentations=logits,
                segmentation_z0_coord=patch_locs[:, 0],
                segmentation_y0_coord=patch_locs[:, 1],
                segmentation_x0_coord=patch_locs[:, 2],
                segmentation_is_valid=instance_mask,
            ),
        )
