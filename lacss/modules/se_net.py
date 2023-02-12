import jax
import treex as tx
jnp = jax.numpy

class ChannelAttention(tx.Module):
    def __init__(
        self,
        squeeze_factor: int = 16,
    ):
        super().__init__()
        self.squeeze_factor = squeeze_factor

    @tx.compact
    def __call__(
        self, 
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        orig_shape = x.shape

        x = x.reshape(-1, orig_shape[-1])
        x_backup = x

        x = jnp.stack([x.max(axis=0), x.mean(axis=0)])

        se_channels = orig_shape[-1] // self.squeeze_factor
        x = tx.Linear(se_channels)(x)
        x = jax.nn.relu(x)
        x = x[0] + x[1]

        x = tx.Linear(orig_shape[-1])(x)
        x = jax.nn.sigmoid(x)

        x = x_backup * x
        x = x.reshape(orig_shape)

        return x

class SpatialAttention(tx.Module):
    def __init__(
        self, 
        filter_size: int = 7,
    ):
        super().__init__()
        self.filter_size = filter_size

    @tx.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
    ) -> jnp.ndarray:

        y = jnp.stack([x.max(axis=-1), x.mean(axis=-1)], axis=-1)
        y = tx.Conv(1, [self.filter_size, self.filter_size])(y)
        y = jnp.nn.sigmoid(y)

        y = x * y

        return y
