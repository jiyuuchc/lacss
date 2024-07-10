from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Sequence, Iterable

import flax.linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp
import numpy as np

from ..typing import Array, ArrayLike

# Inputs are PRNGKey, input shape and dtype.
Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]

class ChannelAttention(nn.Module):
    squeeze_factor: int = 16
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
    ) -> Array:

        orig_shape = x.shape

        x = x.reshape((-1,) + orig_shape[-3:])
        x_backup = x

        x = jnp.stack([x.max(axis=(1,2), keepdims=True), x.mean(axis=(1,2), keepdims=True)])

        se_channels = orig_shape[-1] // self.squeeze_factor
        x = nn.Dense(se_channels, dtype=self.dtype)(x)
        x = jax.nn.relu(x)
        x = x[0] + x[1]

        x = nn.Dense(orig_shape[-1], dtype=self.dtype)(x)
        x = jax.nn.sigmoid(x)

        x = x_backup * x
        x = x.reshape(orig_shape)

        return x


class SpatialAttention(nn.Module):
    filter_size: int = 7
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
    ) -> Array:

        y = jnp.stack([x.max(axis=-1), x.mean(axis=-1)], axis=-1)
        y = nn.Conv(1, [self.filter_size, self.filter_size], dtype=self.dtype)(y)
        y = jax.nn.sigmoid(y)

        y = x * y

        return y


class FPN(nn.Module):
    out_channels: int = 384

    @nn.compact
    def __call__(
        self, inputs: Sequence[ArrayLike], *, training=None
    ) -> Sequence[Array]:
        out_channels = self.out_channels

        outputs = [jax.nn.relu(nn.Dense(out_channels)(x)) for x in inputs]

        for k in range(len(outputs) - 1, 0, -1):
            x = jax.image.resize(outputs[k], outputs[k - 1].shape, "nearest")
            x += outputs[k - 1]
            x = nn.Conv(out_channels, (3, 3))(x)
            x = jax.nn.relu(x)
            outputs[k - 1] = x

        return outputs


class ConvIntegrator(nn.Module):
    """A UNet-like feature integrator.

    Unlike FPN, its feature spaces are not consistent at different spatial scales.
    high-res scale should have lower number of features (like unet)
    """

    n_features: Sequence[int] | None = None
    n_layers: int = 1
    norm: Callable = partial(nn.LayerNorm, use_scale=False)
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: Sequence[Array], *, training=None):
        n_features = self.n_features or [f.shape[-1] for f in x]

        y = x[-1]
        for _ in range(self.n_layers):
            y = nn.Conv(n_features[-1], (3, 3))(y)
            y = self.activation(y)
        output = [y]

        for k in range(-1, -1 * len(x), -1):
            y = nn.ConvTranspose(
                x[k - 1].shape[-1],
                (3, 3),
                strides=(2, 2),
            )(output[0])

            y = self.norm()(x[k - 1] + y)

            for _ in range(self.n_layers):
                y = nn.Conv(n_features[k - 1], (3, 3))(y)
                y = self.activation(y)

            output.insert(0, y)

        return output


class FFN(nn.Module):
    """A feed-forward block commonly used in transformer"""

    dim: int|None = None
    dropout_rate: float = 0.0
    deterministic: bool = False

    @nn.compact
    def __call__(self, x, *, deterministic=None):
        deterministic = deterministic or self.deterministic
        dim = self.dim or x.shape[-1] * 4

        shortcut = x

        x = nn.LayerNorm()(x)

        x = nn.Dense(dim)(x)
        x = jax.nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(shortcut.shape[-1])(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)

        x = shortcut + x

        return x


class Residual(nn.Module):
    """Residual connection module.

    Attributes:
      residual_type: str; residual connection type. Possible values are [
        'gated', 'sigtanh', 'rezero', 'highway', 'add'].
      dtype: Jax dtype; The dtype of the computation (default: float32).
    """

    residual_type: str = 'add'
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, y):
        """Applies the residual connection on given input/output of a module.

        Args:
          x: Input of the module.
          y: Output of the module.

        Returns:
          Output: A combination of the x and y.
        """
        if x.shape != y.shape:
            raise ValueError('x and y should be of the same shape.')

        dtype = self.dtype

        if self.residual_type == 'add':
            return x + y

        elif self.residual_type == 'highway':
            features = x.shape[-1]
            hw_gate = nn.sigmoid(
                nn.Dense(
                    features=features,
                    use_bias=True,
                    kernel_init=initializers.zeros,
                    bias_init=lambda rng, shape, *_: jnp.full(shape, -10.0),
                    dtype=dtype)(x))
            output = jnp.multiply((1 - hw_gate), x) + jnp.multiply(hw_gate, y)

        elif self.residual_type == 'rezero':
            # Based on https://arxiv.org/pdf/2003.04887v1.pdf.
            alpha = self.param('rezero_alpha', initializers.zeros, (1,))
            return x + (alpha * y)

        elif self.residual_type == 'sigtanh':
            # Based on https://arxiv.org/pdf/1606.05328.pdf.
            features = x.shape[-1]
            # sigmoid(W_g.y).
            sigmoid_y = nn.sigmoid(
                nn.Dense(
                    features=features,
                    use_bias=True,
                    kernel_init=initializers.zeros,
                    bias_init=lambda rng, shape, *_: jnp.full(shape, -10.0),
                    dtype=dtype)(y))
            # tanh(U_g.y).
            tanh_y = nn.tanh(
                nn.Dense(
                    features=features,
                    use_bias=False,
                    kernel_init=initializers.zeros,
                    bias_init=initializers.zeros,
                    dtype=dtype)(y))
            return x + (sigmoid_y * tanh_y)

        elif self.residual_type == 'gated':
            # Based on https://arxiv.org/pdf/1910.06764.pdf.
            features = x.shape[-1]
            # Reset gate: r = sigmoid(W_r.x + U_r.y).
            r = nn.sigmoid(
                nn.Dense(
                    features=features,
                    use_bias=False,
                    kernel_init=initializers.zeros,
                    bias_init=initializers.zeros,
                    dtype=dtype)(x) + nn.Dense(
                        features=features,
                        use_bias=False,
                        kernel_init=initializers.zeros,
                        bias_init=initializers.zeros,
                        dtype=dtype)(y))
            # Update gate: z = sigmoid(W_z.x + U_z.y - b_g).
            # NOTE: the paper claims best initializtion for their task for b is 2.
            b_g = self.param('b_g',
                             lambda rng, shape, *_: jnp.full(shape, 10.0),
                             (features,)).astype(dtype)
            z = nn.sigmoid(
                nn.Dense(
                    features=features,
                    use_bias=False,
                    kernel_init=initializers.zeros,
                    bias_init=initializers.zeros,
                    dtype=dtype)(x) + nn.Dense(
                        features=features,
                        use_bias=False,
                        kernel_init=initializers.zeros,
                        bias_init=initializers.zeros,
                        dtype=dtype)(y) - b_g)
            # Candidate_activation: h' = tanh(W_g.y + U_g.(r*x)).
            h = jnp.tanh(
                nn.Dense(
                    features=features,
                    use_bias=False,
                    kernel_init=initializers.zeros,
                    bias_init=initializers.zeros,
                    dtype=dtype)(y) + nn.Dense(
                        features=features,
                        use_bias=False,
                        kernel_init=initializers.zeros,
                        bias_init=initializers.zeros,
                        dtype=dtype)(jnp.multiply(r, x)))

            # Output: g = (1-z)*x + z*h.
            output = jnp.multiply((1.0 - z), x) + jnp.multiply(z, h)

        else:
            raise ValueError(
                f'Residual type {self.residual_type} is not defined.')
        return output


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

        if self.learned_embeddings:
            pos_emb = self.params("pos_emb", self.posemb_init, (embedding_length, dim))
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
