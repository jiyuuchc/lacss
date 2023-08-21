from __future__ import annotations

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..typing import ArrayLike, DataDict
from .common import *
from .unet import UNet


class LacssCollaborator(nn.Module):
    """Collaborator module for semi-supervised Lacss training

    Attributes:
        conv_spec: conv-net specificaiton for cell border predicition
        unet_spec: specification for unet, used to predict cell foreground
        patch_size: patch size for the unet
        n_cls: number of classes (cell types) of input images
    """

    conv_spec: Sequence[int] = (32, 32)
    unet_spec: Sequence[int] = (16, 32, 64)
    patch_size: int = 1
    n_cls: int = 1

    @nn.compact
    def __call__(
        self, image: ArrayLike, cls_id: Optional[ArrayLike] = None
    ) -> DataDict:
        assert cls_id is not None or self.n_cls == 1
        c = cls_id.astype(int).squeeze() if cls_id is not None else 0

        net = UNet(self.conv_spec, self.patch_size)
        _, unet_out = net(image)

        y = unet_out[str(net.start_level)]

        fg = nn.Conv(self.n_cls, (3, 3))(y)
        fg = fg[..., c]

        if fg.shape != image.shape[:-1]:
            fg = jax.image.resize(fg, image.shape[:-1], "linear")

        y = image
        for n_features in self.unet_spec:
            y = nn.Conv(n_features, (3, 3), use_bias=False)(y)
            y = nn.GroupNorm(num_groups=None, group_size=1, use_scale=False)(
                y[None, ...]
            )[0]
            y = jax.nn.relu(y)

        y = nn.Conv(self.n_cls, (3, 3))(y)
        cb = y[..., c]

        return dict(
            fg_pred=fg,
            edge_pred=cb,
        )
