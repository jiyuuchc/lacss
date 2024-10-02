"""Vision Transformer.
Based on scenic implementation: https://github.com/google-research/scenic/
"""

from typing import Any, Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .common import DropPath, IdentityLayer, FFN, PositionEmbedding1D, PositionEmbedding2D
from ..typing import Array, ArrayLike

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]

class MAPHead(nn.Module):
    """Multihead Attention Pooling."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-3] * x_shape[-2], x_shape[-1])
        probe = self.param(
            'probe', 
            nn.initializers.xavier_uniform(), 
            (1, 1, x.shape[-1]),
            self.param_dtype,
        )
        probe = probe.astype(x.dtype)
        probe = jnp.tile(probe, [x.shape[0], 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm()(x)
        x = x + FFN(
            self.mlp_dim,
            dropout_rate=0.0,
        )(y, deterministic=True)

        return x[:, 0].reshape(*x_shape[:-3], -1)


class VitEncoder(nn.Module):
    """Transformer Encoder.

    Attributes:
      n_layers: Number of layers.
      n_heads: The number of heads for multi-head self-attention.
      dim_ffn: Dimension of the mlp on top of attention block.
      pos_embedding_type: The type of positional embeddings to add to the
        input tokens.
      dropout: Dropout rate.
      stochastic_depth: probability of dropping a layer linearly grows
        from 0 to the provided value. Our implementation of stochastic depth
        follows timm library, which does per-example layer dropping and uses
        independent dropping patterns for each skip-connection.
    """
    n_layers: int = 6
    n_heads: int = 12
    dim_ffn: int|None = None
    pos_embedding_type: str = 'learned_1d'
    dropout: float = 0.1
    attention_dropout: float = 0.1
    stochastic_depth: float = 0.0

    @nn.compact
    def __call__(self, inputs: ArrayLike, cls_token: ArrayLike|None=None, *, training: bool = False):
        """Applies Transformer model on the inputs.

        Args:
          inputs: Input tokens of shape [..., num_tokens, channels].
          cls_token: addtional class token if not None
          training: If in training mode, dropout and stochastic depth is applied.

        Returns:
          Encoded tokens and cls_token
        """

        input_shape = inputs.shape
        H, W, C = input_shape[-3:]

        x = inputs.reshape((-1,) + (H*W, C))

        if self.pos_embedding_type == 'learned_1d':
            x = x + PositionEmbedding1D()(x)

        elif self.pos_embedding_type == 'sinusoidal_2d':
            pos = PositionEmbedding2D(
                posemb_init=None,
            )(input_shape)
            x = x + pos.reshape(-1, C)

        elif self.pos_embedding_type == 'learned_2d':
            pos = PositionEmbedding2D()(input_shape)
            x = x + pos.reshape(-1, C)
        
        elif self.pos_embedding_type != 'none':
            raise ValueError("invalid pos_embedding_type {self.pos_embedding_type}")

        if cls_token:
            assert cls_token.shape == input_shape[:-3] + (C,), f"invalid class token shape {cls_token.shape}"
            x = jnp.c_[x, cls_token.reshape(-1, 1, C)]

        # x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)

        for lyr in range(self.n_layers):
            shortcut = x
            x = nn.Layernorm()(x)
            x = nn.MultiHeadAttention(
                num_heads=self.n_heads,
                dropout_rate=self.attention_dropout,
                kernel_init=nn.initializers.xavier_uniform(),
                broadcast_dropout=False,
                name=f"attention_{lyr}",
            )(x, deterministic=not training)
            
            x = FFN(dim=self.dim_ffn, dropout_rate=self.dropout, name="ffn_{lyr}")(
                x, deterministic=not training
            )

            droppath_rate = lyr/self.n_layers*self.stochastic_depth
            x = DropPath(droppath_rate)(x, deterministic=not training)

            x = shortcut + x

        if cls_token:
            cls_token = x[..., -1, :].reshape(input_shape[:-3] + (C,))
            x = x[..., :-1, :].reshape(input_shape)
            return x, cls_token
        else:            
            x = x.reshape(input_shape)
            return x, None


class ViT(nn.Module):
    """Vision Transformer model.

      Attributes:
        dim: Size of the hidden state of the output of model's stem.
        dim_ffn: Dimension of the mlp on top of attention block.
        n_layers: Number of layers.
        n_heads: Number of self-attention heads.
        patch_size: specify the patch dimension for tokenization
        pos_embedding_type: The type of positional embeddings to add to the
            tokens at the beginning of the transformer encoder. 
        dropout: Dropout rate.
        attention_dropout: Dropout for attention heads.

        n_cls: Number of output classes.
        representation_size: Size of the representation layer in the model's head.
            if None, we skip the extra projection + tanh activation at the end.
        classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
            'token', or None to return encoder features
    """
    # transformer parameters
    dim: int = 256
    dim_ffn: int|None = None
    n_layers: int = 8
    n_heads: int = 12
    patch_size: tuple[int,int] = (8, 8)
    pos_embedding_type: str = 'learned_1d'
    dropout: float = 0.1
    attention_dropout: float = 0.1
    stochastic_depth: float = 0.0

    # classifier parameters
    n_cls: int = -1
    representation_size: Optional[int] = None
    classifier_type: str|None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, training: bool=False):
        """ViT model

        Args:
          inputs: Input [..., H, W, C].
          training: If in training mode, dropout and stochastic depth is applied.

        Returns:
          features: [..., H//patch_size, W//patch_size, n_dim_hidden]
        """
        input_shape = x.shape

        x = x.reshape((-1,) + input_shape[-3:]) # [b, h, w, c]

        fh, fw = self.patch_size
        x = nn.Conv(
            self.dim, (fh, fw),
            strides=(fh, fw),
            padding='VALID',
            name='embedding',
        )(x)

        output_shape = x.shape

        # x = jnp.reshape(x.shape[0], -1, x.shape[-1]) # merge H and W

        # Add a class token if suitable
        if self.classifier_type == 'token':
            cls_token = self.param('cls', nn.initializers.zeros, (1, x.shape[-1]), self.param_dtype)
            cls_toekn = cls_token.astype(x.dtype)
            cls_token = jnp.tile(cls_token, [x.shape[0], 1])
        else:
            cls_token = None

        x, cls_token = VitEncoder(
            dim_ffn=self.dim_ffn,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            pos_embedding_type=self.pos_embedding_type,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            stochastic_depth=self.stochastic_depth,
            name='Transformer',
        )(x, cls_token, training=training)

        if self.classifier_type is None:
            return x.reshape(output_shape)

        if self.classifier_type in ('token', '0'):
            x = cls_token

        elif self.classifier_type in ('gap', 'gmp', 'gsp'):
            fn = {'gap': jnp.mean, 'gmp': jnp.max,
                  'gsp': jnp.sum}[self.classifier_type]
            x = fn(x, axis=1)

        elif self.classifier_type == 'map':
            x = MAPHead(
                num_heads=self.n_heads, 
                mlp_dim=self.dim_ffn, 
            )(x)

        else:
            raise ValueError(f'Unknown classifier {self.classifier_type}')

        if self.representation_size is not None:
            x = nn.Dense(self.representation_size, name='pre_logits')(x)
            x = nn.tanh(x)
        else:
            x = IdentityLayer(name='pre_logits')(x)

        # If self.n_cls <= 0, we just return the representations
        if self.n_cls > 0:
            x = nn.Dense(
                self.n_cls,
                kernel_init=nn.initializers.zeros,
                name='output_projection',
            )(x)

        output = x.reshape(input_shape[:-3] + x.shape[-1:])

        return output
