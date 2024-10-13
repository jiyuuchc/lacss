from __future__ import annotations

from typing import Optional, Sequence, Callable

import flax.linen as nn
import jax.numpy as jnp

from .convnext import ConvNeXt
from .integrators import FPN
from ..typing import DataDict, Array, ArrayLike, Any
from .common import DefaultUnpicklerMixin
from .video_integrator import VideoIntegrator

class Backbone(nn.Module, DefaultUnpicklerMixin):
    base_type: str = "tiny"
    patch_size: int = 4
    fpn_dim: int = 384
    out_dim: int = 256
    det_layer: int = 0
    seg_layer: int = 0
    n_layers_det: int = 4
    n_layers_seg: int = 3
    drop_path_rate: float=0.4
    activation: Callable[[ArrayLike], Array]=nn.gelu
    deterministic: bool|None = None
    dtype: Any = None

    def setup(self):
        self.cnn = ConvNeXt.get_preconfigured(
            self.base_type, 
            patch_size=self.patch_size, 
            drop_path_rate = self.drop_path_rate, 
            dtype=self.dtype
        )

    def get_ref(self, x):
        if x.ndim == 3:
            x = x[None, ...]
            return self.cnn(x, deterministic=True)

        else:
            assert x.ndim == 4

            to_shape = (-1, self.patch_size) + x.shape[1:]
            image_ = x
            image_ = image_.reshape(to_shape).mean(axis=1)
            f0 = self.cnn(image_, deterministic=True)

            image_ = x.swapaxes(0,1)
            image_ = image_.reshape(to_shape).mean(axis=1)
            f1 = self.cnn(image_, deterministic=True)

            image_ = x.swapaxes(0,2)
            image_ = image_.reshape(to_shape).mean(axis=1)
            f2 = self.cnn(image_, deterministic=True)

            return (f0, f1, f2)

    @nn.compact
    def get_2d_feature(self, image, video_refs, deterministic):
        x = jnp.asarray(image)
        if image.ndim == 3:
            x = x[None, ...]

        assert x.ndim == 4

        x = self.cnn(x, deterministic=deterministic or video_refs is not None)
        self.sow("intermediates", "encoder", x)

        if video_refs is not None:
            x = VideoIntegrator(dtype=self.dtype)(x, video_refs, deterministic=deterministic)

        x = FPN(self.fpn_dim, activation=self.activation, dtype=self.dtype)(x)

        self.sow("intermediates", "decoder", x)

        x_det = x[self.det_layer]
        for _ in range(self.n_layers_det):
            y = nn.Conv(self.out_dim, (3,3), dtype=self.dtype)(x_det)
            y = nn.GroupNorm(dtype=self.dtype)(y)
            x_det = nn.gelu(y)

        x_seg = x[self.seg_layer]
        for _ in range(self.n_layers_seg):
            y = nn.Conv(self.out_dim, (3,3), dtype=self.dtype)(x_seg)
            y = nn.GroupNorm(dtype=self.dtype)(y)
            x_seg = nn.gelu(y)

        if image.ndim == 3:
            x_det = x_det.squeeze(0)
            x_seg = x_seg.squeeze(0)

        return x_det, x_seg


    def __call__(self, image:ArrayLike, video_refs:tuple|None=None, deterministic:bool|None=True):
        image = jnp.asarray(image)
        if deterministic is None:
            deterministic = self.deterministic

        if image.ndim == 3:
            return self.get_2d_feature(image, video_refs, deterministic)

        else:
            def _get_ref(vf, axis):
                if vf is None:
                    return None
                else:
                    return (vf[0][axis], vf[1])

            img_shape = image.shape
            image_ = image
            image_ = image_.reshape((-1, self.patch_size) + img_shape[1:]).mean(axis=1)
            ref_features = _get_ref(video_refs, 0)
            f0 = self.get_2d_feature(image_, ref_features, deterministic)

            image_ = image.swapaxes(0,1)
            image_ = image_.reshape((-1, self.patch_size) + img_shape[1:]).mean(axis=1)
            ref_features = _get_ref(video_refs, 1)
            f1 = self.get_2d_feature(image_, ref_features, deterministic)

            image_ = image.swapaxes(0,2)
            image_ = image_.reshape((-1, self.patch_size) + img_shape[1:]).mean(axis=1)
            ref_features = _get_ref(video_refs, 2)            
            f2 = self.get_2d_feature(image_, ref_features, deterministic)

            x_det = jnp.c_[f0[0], f1[0].swapaxes(0,1), f2[0].swapaxes(0,2)]
            x_seg = jnp.c_[f0[1], f1[1].swapaxes(0,1), f2[1].swapaxes(0,2)]

            return x_det, x_seg
    