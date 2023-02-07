import jax
import treex as tx
jnp = jax.numpy

class AuxNet(tx.Module):
    n_groups = 1
    conv_spec = (24,64,64)
    share_weights = None

    def __init__(self, conv_spec=(24,64,64), n_groups=1, share_weights=False):
        self.conv_spec = conv_spec
        self.n_groups = n_groups
        self.share_weights = share_weights

    @tx.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for n in self.conv_spec:
            # x = tx.Conv(n, (3,3))(x)
            x = tx.Conv(n, (3,3), use_bias=False)(x)
            x = tx.GroupNorm(num_groups=None, group_size=1, use_scale=False)(x)
            x = jax.nn.relu(x)

        x = tx.Conv(self.n_groups, (3,3))(x)
        x = jax.nn.sigmoid(x)

        return x

    def get_config(self):
        return dict(
            conv_spec = self.conv_spec,
            n_groups = self.n_groups,
            share_weights = self.share_weights,
        )

    @classmethod
    def from_config(cls, config):
        return(cls(**config))
