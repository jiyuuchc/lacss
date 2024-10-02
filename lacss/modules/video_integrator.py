from __future__ import annotations

import flax.linen as nn
import jax
from .common import FFN, PositionEmbedding1D
from ..typing import Any, Sequence, Array, ArrayLike
jnp = jax.numpy

class VideoIntegrator(nn.Module):
    n_heads: int = 4
    dropout: float = 0.2
    ff_dropout: float = 0.2
    dtype: Any = None

    @nn.compact
    def __call__(self, feature:Sequence[ArrayLike], video_refs:tuple[Sequence[ArrayLike], ArrayLike|None], deterministic:bool=True)->Sequence[Array]:
        ref_feature, ref_mask = video_refs

        out = []
        for k in range(4):
            x = jnp.asarray(feature[k])[..., None, :]

            y = jnp.moveaxis(ref_feature[k], 0, -2)
            pos_embedding = PositionEmbedding1D()(y)

            x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)(x)
            x = nn.MultiHeadDotProductAttention(
                self.n_heads,
                dropout_rate=self.dropout,
                dtype=self.dtype,
            )(
                inputs_q = x,
                inputs_k = y + pos_embedding,
                inputs_v = y,
                deterministic=deterministic,
                sow_weights=True,
                mask=ref_mask,
            )

            x = FFN(dropout_rate=self.ff_dropout, dtype=self.dtype)(x, deterministic=deterministic)

            x = x.squeeze(-2)

            out.append(feature[k] + x)

        return out
