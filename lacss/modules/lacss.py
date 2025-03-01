from __future__ import annotations

from dataclasses import field
from functools import partial

import flax.linen as nn
import jax

from ..typing import Any, Array, ArrayLike
from ..utils import deep_update
from .backbone import Backbone
from .common import DefaultUnpicklerMixin
from .lpn import LPN
from .lpn3d import LPN3D
from .segmentor import Segmentor
from .segmentor3d import Segmentor3D

jnp = jax.numpy


class Lacss(nn.Module, DefaultUnpicklerMixin):
    """Main class for LACSS model

    Attributes:
        backbone: CNN backbone
        detector: Detection head to predict cell locations
        segmentor: Segmentation head
        detector_3d: Detection head for 3d input
        segmentor_3d: Segmentation head for 3d input
    """

    backbone: nn.Module = field(default_factory=Backbone)
    detector: nn.Module = field(default_factory=LPN)
    segmentor: nn.Module | None = field(default_factory=Segmentor)
    detector_3d: nn.Module | None = None
    segmentor_3d: nn.Module | None = None

    def get_features(self, image, ref_features, deterministic):
        return self.backbone(image, ref_features, deterministic=deterministic)

    def __call__(
        self,
        image: ArrayLike,
        *,
        image_mask: ArrayLike | None = None,
        video_refs: tuple | None = None,
    ) -> dict:
        """
        Args:
            image: [H, W, C] or [D, H, W, C]
        Returns:
            a dict of model outputs

        """
        # add a batch dim if needed
        image = jnp.asarray(image)

        x_det, x_seg = self.get_features(image, video_refs, deterministic=True)
        outputs = dict(
            det_features=x_det,
            seg_features=x_seg,
        )

        if image.ndim == 3:
            detector_outputs = self.detector(x_det, mask=image_mask)
            outputs = deep_update(outputs, detector_outputs)

            if self.segmentor is not None:
                segmentor_out = self.segmentor(
                    x_seg,
                    outputs["predictions"]["locations"],
                )
                outputs = deep_update(outputs, segmentor_out)

        else:
            if self.detector_3d is None:
                raise (ValueError("not a 3d detector"))

            detector_outputs = self.detector_3d(x_det, mask=image_mask)
            outputs = deep_update(outputs, detector_outputs)

            if self.segmentor_3d is not None:
                # segmentor_out = self.segmentor_3d(
                # x_seg, outputs["predictions"]["locations"],
                # )

                locs = outputs["predictions"]["locations"]
                locs = locs.reshape((locs.shape[0] // 16, 16) + locs.shape[1:])
                _, segmentor_out = jax.lax.scan(
                    lambda _, locs_: (None, self.segmentor_3d(x_seg, locs_)),
                    None,
                    locs,
                )
                segmentor_out = jax.tree_util.tree_map(
                    lambda data: data.reshape(-1, *data.shape[2:]),
                    segmentor_out,
                )

                outputs = deep_update(outputs, segmentor_out)

        return outputs

    def get_config(self):
        from dataclasses import asdict

        from ..utils import remove_dictkey

        cfg = asdict(self)

        remove_dictkey(cfg, "parent")

        return cfg

    @classmethod
    def from_config(cls, config):
        def _build_sub_module(module, k):
            if k in config and config[k] is not None:
                return module(**config[k])
            else:
                return None

        if isinstance(config, str):
            return cls.get_preconfigued(config)

        else:
            backbone = _build_sub_module(Backbone, "backbone")
            detector = _build_sub_module(LPN, "detector")

            assert backbone is not None and detector is not None, f"invalid config"

            obj = Lacss(
                backbone=backbone,
                detector=detector,
                segmentor=_build_sub_module(Segmentor, "segmentor"),
                detector_3d=_build_sub_module(LPN3D, "detector_3d"),
                segmentor_3d=_build_sub_module(Segmentor3D, "segmentor_3d"),
            )

            return obj

    @classmethod
    def get_default_model(cls, dtype=None):
        return cls.get_base_model(dtype=dtype)

    @classmethod
    def get_small_model(cls, dtype=None):
        return cls(
            backbone=Backbone(
                "tiny",
                fpn_dim=384,
                out_dim=256,
                dtype=dtype,
            ),
            detector=LPN(dtype=dtype),
            segmentor=Segmentor(
                patch_dim=32,
                sig_dim=512,
                pos_emb_shape=(16, 16, 4),
                dtype=dtype,
            ),
            detector_3d=LPN3D(dim=384, dtype=dtype),
            segmentor_3d=Segmentor3D(
                patch_dim=48,
                sig_dim=1024,
                pos_emb_shape=(8, 8, 8, 4),
                dtype=dtype,
            ),
        )

    @classmethod
    def get_base_model(cls, dtype=None):
        return cls(
            backbone=Backbone("base_v2", fpn_dim=512, out_dim=384, dtype=dtype),
            detector=LPN(dtype=dtype),
            segmentor=Segmentor(
                patch_dim=64,
                sig_dim=768,
                pos_emb_shape=(16, 16, 6),
                dtype=dtype,
            ),
            detector_3d=LPN3D(dim=512, dtype=dtype),
            segmentor_3d=Segmentor3D(
                patch_dim=64,
                sig_dim=1384,
                pos_emb_shape=(8, 8, 8, 6),
                dtype=dtype,
            ),
        )

    @classmethod
    def get_large_model(cls, dtype=None):
        return cls(
            backbone=Backbone("large_v2", fpn_dim=768, out_dim=512, dtype=dtype),
            detector=LPN(dtype=dtype),
            segmentor=Segmentor(
                patch_dim=96,
                sig_dim=1024,
                pos_emb_shape=(16, 16, 8),
                dtype=dtype,
            ),
            detector_3d=LPN3D(dim=768, dtype=dtype),
            segmentor_3d=Segmentor3D(
                patch_dim=96,
                sig_dim=2048,
                pos_emb_shape=(8, 8, 8, 8),
                dtype=dtype,
            ),
        )

    @classmethod
    def get_preconfigued(cls, config: str, dtype=None):
        if config == "default" or config == "base":
            return cls.get_default_model(dtype=dtype)
        elif config == "small":
            return cls.get_small_model(dtype=dtype)
        elif config == "large":
            return cls.get_large_model(dtype=dtype)
        else:
            raise ValueError(f"Unkown model config {config}")
