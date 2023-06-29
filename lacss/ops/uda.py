import jax


@jax.custom_vjp
def gradient_reversal(x):
    return x


def _gr_fwd(x):
    return x, None


def _gr_bwd(_, g):
    return (jax.tree_util.tree_map(lambda v: -v, g),)


gradient_reversal.defvjp(_gr_fwd, _gr_bwd)
""" A gradient reveral op. 

    This is a no-op during inference. During training, it does nothing
    in the forward pass, but reverse the sign of gradient in backward
    pass. Typically placed right before a discriminator in adversal 
    training.
"""
