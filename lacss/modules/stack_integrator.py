from __future__ import annotations

from typing import Sequence, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ..typing import ArrayLike, Array
from .common import FPN, FFN

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]

class _StackTransformer(nn.Module):
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0. 
    ff_dropout: float = 0.
    deterministic: bool|None = None
    leared_embedding: bool = False

    def _get_pos_embeddings(self, feature):
        max_embedding_length = 32
        depth, height, width, n_ch = feature.shape

        if self.leared_embedding:
            pos_embedding = self.param(
                'pos_embedding', 
                nn.initializers.normal(stddev=0.02), 
                (max_embedding_length, n_ch),
                feature.dtype,
            )
        else:
            pos_embedding = np.zeros((max_embedding_length, n_ch))
            pos = np.arange(max_embedding_length) - max_embedding_length//2
            pos = pos[:, None] * np.exp(- np.log(64) / n_ch * np.arange(0, n_ch, 2))
            pos_embedding[:, 0::2] = np.sin(pos)
            pos_embedding[:, 1::2] = np.cos(pos)

        pos_embedding = pos_embedding[max_embedding_length//2+(-depth)//2:max_embedding_length//2+depth//2]

        assert pos_embedding.shape == (depth, n_ch)

        return jnp.asarray(pos_embedding)

    def _attention(self, feature, pos_embedding, deterministic):
        shortcut = feature

        feature = nn.LayerNorm()(feature)
        feature = nn.MultiHeadDotProductAttention(
            self.n_heads,
            dropout_rate=self.dropout,
            broadcast_dropout=False,
        )(
            inputs_q = feature + pos_embedding,
            inputs_k = feature + pos_embedding,
            inputs_v = feature,
            deterministic=deterministic,
            sow_weights=True,
        )
        feature = shortcut + feature

        return feature

    @nn.compact
    def __call__(self, feature:ArrayLike, deterministic:bool|None=None) -> Array:
        assert feature.ndim == 4, f"Wrong input dimension to StackIntegrator: {feature.shape}"

        if deterministic is None:
            deterministic = self.deterministic

        # if feature.shape[0] == 1:
        #     return feature

        pos_embedding = self._get_pos_embeddings(feature)

        feature = jnp.moveaxis(feature, 0, 2) # we are integrating along z axis
        for _ in range(self.n_layers):
            feature = self._attention(feature, pos_embedding, deterministic=deterministic)
            feature = FFN(dropout_rate=self.ff_dropout)(feature, deterministic=deterministic)        
        feature = jnp.moveaxis(feature, 2, 0) # move the z axis back

        return feature


class StackIntegrator(nn.Module):
    """ Feature integration for image stacks.

    This applies a simple transformer on the stack of features at the lowest resolution to integrate information
    from all inputs images. This computation is very light-weight, becasue the integration happens only for 
    features at the same location -- the y-x axis are treated as batch dimension. After transformer, the features
    at different resolutions are mixed using a standard FPN.

    Attributes:
      n_layers: number transformer layers
      n_heads: number of heads

      dropout: transformer attention weight dropout rate
      ff_dropout: the dropout rate of the FFN part of the transformer

      dim_out: the final output feature dim (after FPN)
    """
    n_layers: int = 2
    n_heads: int = 4
    dim_out: int = 384
    dropout: float = 0.
    ff_dropout: float = 0.

    @nn.compact
    def __call__(self, features:Sequence[ArrayLike], *, training:bool=False) -> Sequence[Array]:
        feature_last = features[-1]
        feature_last = _StackTransformer(
            self.n_layers,
            self.n_heads,
            self.dropout,
            self.ff_dropout,
        )(
            feature_last, deterministic=not training,
        )

        features = list(features[:-1]) + [feature_last]

        return FPN(self.dim_out)(features, training=training)
