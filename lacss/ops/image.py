from functools import partial
import jax
from jax.scipy.signal import convolve
jnp = jax.numpy

def sorbel_edges(image):
    """Returns a tensor holding Sobel edge maps.
    >>> image = random.uniform(key, shape=[3, 28, 28])
    >>> sobel = sobel_edges(image)
    >>> sobel_y = sobel[0, :, :, :] # sobel in y-direction
    >>> sobel_x = sobel[1, :, :, :] # sobel in x-direction
    ```
    Args:
        image: [n, h, w]
    Returns:
        Tensor holding edge maps for each channel. [2, n, h, w] 
    """

    kernels = jnp.array(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    )

    sorbel_filter = jax.vmap(
        jax.vmap(
            partial(convolve, mode='valid'),
            in_axes = [0, None],
        ),
        in_axes = [None, 0],
    )

    image = jnp.pad(image, [[0,0], [1,1], [1,1]], mode='reflect')
    output = sorbel_filter(image, kernels)

    return output
