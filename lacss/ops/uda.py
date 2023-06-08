import jax


@jax.custom_vjp
def gradient_reversal(x):
    return x


def _gr_fwd(x):
    return x, None


def _gr_bwd(_, g):
    return (jax.tree_util.tree_map(lambda v: -v, g),)


gradient_reversal.defvjp(_gr_fwd, _gr_bwd)
