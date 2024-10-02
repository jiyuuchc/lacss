from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Sequence, Iterable, Any

import flax.linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp
import numpy as np

from ..typing import Array, ArrayLike

# Inputs are PRNGKey, input shape and dtype.
Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]

import warnings

def picklable_relu(x):
    return jax.nn.relu(x)

class DefaultUnpicklerMixin:
    """Provides an unpickling method  which will restore
    module instances, even if the model has gained new fields
    or dropped some fields since data was unpickled.
    """
    def __setstate__(self, state):
        try:
            current_fields = self.__class__.__dataclass_fields__
        except AttributeError:
            warnings.warn("DefaultUnpickler used in a class that is not a dataclass")
            current_fields = {}
        for key in list(state.keys()):
            if key not in current_fields:
                if not key[0] == "_":
                    warnings.warn(f"Dropping unused field {key}")
                del state[key]
        for key in current_fields:
            if key not in state:
                if not key == "parent":
                    warnings.warn(f"Adding missing field {key}")
                state[key] = current_fields[key].default

        self.__init__(**state)


class _ChannelAttention(nn.Module):
    squeeze_factor: int = 16
    dtype: Any = None

    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
    ) -> Array:

        orig_shape = x.shape
        se_channels = orig_shape[-1] // self.squeeze_factor

        x_backup = x

        x = x.reshape(-1, orig_shape[-1])
        x = jnp.concatenate([x.max(axis=0), x.mean(axis=0)], axis=-1)

        x = nn.Dense(se_channels, dtype=self.dtype)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(orig_shape[-1], dtype=self.dtype)(x)
        x = jax.nn.sigmoid(x)

        x = x_backup * x

        return x


ChannelAttention = nn.vmap(
    _ChannelAttention, 
    variable_axes={"params":None}, 
    split_rngs={"params":False},
)


class SpatialAttention(nn.Module):
    filter_size: int|tuple[int,...] = 7
    dtype: Any = None

    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
    ) -> Array:

        try:
            conv_filter = len(self.filter_size)
        except:
            conv_filter = (self.filter_size, self.filter_size)
        
        y = jnp.stack([x.max(axis=-1), x.mean(axis=-1)], axis=-1)
        y = nn.Conv(1, conv_filter, dtype=self.dtype)(y)
        y = jax.nn.sigmoid(y)

        y = x * y

        return y


class FFN(nn.Module):
    """A feed-forward block commonly used in transformer"""

    dim: int|None = None
    dropout_rate: float = 0.0
    deterministic: bool = False
    dtype: Any = None

    @nn.compact
    def __call__(self, x, *, deterministic=None):
        deterministic = deterministic or self.deterministic
        orig_dim = x.shape[-1]
        dim = self.dim or orig_dim * 4

        x = nn.LayerNorm(dtype=self.dtype)(x)

        x = nn.Dense(dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)

        x = nn.Dense(orig_dim, dtype=self.dtype)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)

        return x


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


def get_constant_initializer(constant: float) -> Initializer:
    """Returns an initializer that initializes everything to a given constant."""

    def init_fn(unused_key: jnp.ndarray,  # pytype: disable=annotation-type-mismatch  # jnp-type
                shape: Iterable[int],
                dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
        return constant * jnp.ones(shape, dtype=dtype)

    return init_fn  # pytype: disable=bad-return-type  # jax-ndarray


class Affine(nn.Module):
    """Affine transformation layer.

    Described in:
    Touvron et al, "ResMLP: Feedforward networks for image classification
    with data-efficient training", 2021.

    Performs an affine transformation on the final dimension of the input tensor.
    """
    bias_init: Initializer = nn.initializers.zeros
    scale_init: Initializer = nn.initializers.ones
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        n = x.shape[-1]
        scale = self.param('scale', self.scale_init, (n,))
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (n,))
        else:
            bias = 0.0
        return scale * x + bias


class DropPath(nn.Module):
    """Performs layer-dropout (also known as stochastic depth).

    Described in
    Huang & Sun et al, "Deep Networks with Stochastic Depth", 2016
    https://arxiv.org/abs/1603.09382

    Attributes:
      rate: the layer dropout probability (_not_ the keep rate!).
      deterministic: If false (e.g. in training) the inputs are scaled by `1 / (1
        - rate)` and the layer dropout is applied, whereas if true (e.g. in
        evaluation), no stochastic depth is applied and the inputs are returned as
        is.
    """
    rate: float = 0.0
    deterministic: Optional[bool] = None
    broadcast_dims: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 deterministic: Optional[bool] = None) -> jnp.ndarray:
        """Applies a stochastic depth mask to the inputs.

        Args:
            x: Input tensor.
            deterministic: If false (e.g. in training) the inputs are scaled by `1 /
                (1 - rate)` and the layer dropout is applied, whereas if true (e.g. in
                evaluation), no stochastic depth is applied and the inputs are returned
                as is.
            broadcast_dims: default assumes the input is a single sample with no batch-axis

        Returns:
            The masked inputs reweighted to preserve mean.
        """
        if self.broadcast_dims is None:
            broadcast_dims = range(x.ndim)
        else:
            broadcast_dims = self.broadcast_dims

        return nn.Dropout(
            rate=self.rate, broadcast_dims=broadcast_dims)(x, deterministic)


class PositionEmbedding1D(nn.Module):
    """ 1D positional embeddings

    Attributes:
      posemb_init: if None, use cosine embedding
      max_len: Maximum possible length for the input. If None, the max_len is
        set to the inputs sequence length.
      max_timescale: time scale for  
      rescale_from: If not None, embeddings are rescaled from this length.
    """
    posemb_init: Initializer|None = nn.initializers.normal(stddev=0.02)
    max_len: int|None = None
    max_timescale: float = 1.0e4
    rescale_from: int|None = None

    @nn.compact
    def __call__(self, input: tuple|ArrayLike) -> Array:
        """
        Args:
          inputs: Input data that needs embedding or shape tuple

        Returns:
          Output: `(Sequence_length, inputs_dim)`.
        """
        if isinstance(input, tuple):
            input_shape = input
        else:
            input_shape = input.shape

        assert len(input_shape) >= 2, f"invalid input shape {input_shape}"
        length, dim = input_shape[-2:]
        max_len = self.max_len or length

        if self.rescale_from: 
            embedding_length = self.rescale_from
        else:
            embedding_length = max_len

        if self.posemb_init is not None:
            pos_emb = self.param("pos_emb", self.posemb_init, (embedding_length, dim))
        else:
            pos_emb = np.zeros((embedding_length, dim), dtype=np.float32)
            position = np.arange(embedding_length)[:, np.newaxis]
            div_term = np.exp(
                np.arange(0, dim, 2) / dim * np.log(self.max_timescale)
            )
            pos_emb[:, 0::2] = np.sin(position / div_term)
            pos_emb[:, 1::2] = np.cos(position / div_term)
            pos_emb = jnp.asarray(pos_emb)

        if self.rescale_from:
            pos_emb = jax.image.resize(
                pos_emb, (max_len, dim), method='bilinear', antialias=False
            )

        pos_emb = pos_emb[:length, :]

        assert pos_emb.shape == (length, dim)

        return pos_emb


class PositionEmbedding2D(nn.Module):
    """2D cosine positional embeddings

    Attributes:
      rescale_from: If not None, embeddings are rescaled from this shape.
      posemb_init: Positional embedding initializer.
    """
    posemb_init: Initializer = nn.initializers.normal(stddev=0.02)
    rescale_from: tuple[int, int]|None = None

    @nn.compact
    def __call__(self, input: tuple|ArrayLike) -> Array:
        """
        Args:
          inputs: Input data that needs embedding or shape tuple

        Returns:
          Output: `(h, w, inputs_dim)`.
        """
        if isinstance(input, tuple):
            input_shape = input
        else:
            input_shape = input.shape

        assert len(input_shape) >= 3, f"invalid input shape {input_shape}"
        h, w, c = input_shape[-3:]

        if self.rescale_from:  # `[h, w, c]`
            embedding_h, embedding_w = self.rescale_from[0], self.rescale_from[1]
        else:
            embedding_h, embedding_w = h, w

        row_pos_embed = self.param('row_pos_embedding', self.posemb_init, (embedding_w, c // 2))
        col_pos_embed = self.param('col_pos_embedding', self.posemb_init, (embedding_h, c // 2))

        # To `[h, w, c//2]`.
        x_pos_emb = jnp.tile(
            jnp.expand_dims(row_pos_embed, axis=0), (embedding_h, 1, 1)
        )
        y_pos_emb = jnp.tile(
            jnp.expand_dims(col_pos_embed, axis=1), (1, embedding_w, 1)
        )

        # To `[h, w, c]`.
        pos = jnp.concatenate((x_pos_emb, y_pos_emb), axis=-1)

        if self.rescale_from:
            pos = jax.image.resize(
                pos, (h, w, c), method='bilinear', antialias=False
            )

        return pos


class GRN(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        gamma = self.param("gamms", nn.initializers.zeros, (1, 1, 1, dim))
        beta = self.param("beta", nn.initializers.zeros, (1, 1, 1, dim))

        x_ = jnp.array(x, "float32")
        mu2 = jax.lax.square(jnp.abs(x_)).mean(axis=(1,2), keepdims=True)
        mu2 = jnp.maximum(mu2, 1e-6)
        Gx = jax.lax.sqrt(mu2).astype(x.dtype)
        Nx = Gx / (Gx.mean(axis=-1, keepdims=True) + self.epsilon)

        return gamma * (x * Nx) + beta + x
