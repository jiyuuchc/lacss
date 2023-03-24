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
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    )

    sorbel_filter = jax.vmap(
        jax.vmap(
            partial(convolve, mode="valid"),
            in_axes=[0, None],
        ),
        in_axes=[None, 0],
    )

    image = jnp.pad(image, [[0, 0], [1, 1], [1, 1]], mode="reflect")
    output = sorbel_filter(image, kernels)

    return output


def _retrieve_value_at(img, loc, out_of_bound_value=0.0):

    iloc = jnp.floor(loc).astype(int)
    res = loc - iloc

    offsets = jnp.asarray(
        [[(i >> j) % 2 for j in range(len(loc))] for i in range(2 ** len(loc))]
    )
    ilocs = tuple(jnp.swapaxis(iloc + offsets, 0, 1))

    weight = jnp.prod(res * (offsets == 0) + (1 - res) * (offsets == 1), axis=1)

    value = (img[ilocs] * weight[:, None]).sum(aixs=0)

    valid = (loc >= 0).all() & ((loc + 1) <= jnp.asarray(img.shape)).all()
    value = jnp.where(valid, value, out_of_bound_value)

    return value


def retrieve_value(img, locs, out_of_bound_value=0.0):
    """Retrieve values as non-integer locations by interpolation
    Args:
        img: [D1,D2,..,Dk, ...]
        locs: [d1,d2,..,dn, k]
        out_of_bound_value: optional float constant, defualt 0.
    Returns:
        values: [d1,d2,..,dn, ...], float
    """

    loc_shape = locs.shape
    img_shape = img.shape
    d_loc = loc_shape[-1]

    img = img.reshape(img_shape[:d_loc] + (-1,))
    locs = locs.reshape(-1, d_loc)
    op = partial(_retrieve_value_at, img=img, out_of_bound_value=out_of_bound_value)

    values = jax.vmap(op)(locs)
    out_shape = loc_shape[:-1] + img_shape[d_loc:]

    values = values.reshape(out_shape)

    return values
