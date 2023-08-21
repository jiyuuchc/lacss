from __future__ import annotations

from functools import partial

import jax
from jax.scipy.signal import convolve

jnp = jax.numpy

from ..typing import *


def sorbel_edges(image: ArrayLike) -> Array:
    """Returns a tensor holding Sobel edge maps.

    Examples:

        >>> image = random.uniform(key, shape=[3, 28, 28])
        >>> sobel = sobel_edges(image)
        >>> sobel_y = sobel[0, :, :, :] # sobel in y-direction
        >>> sobel_x = sobel[1, :, :, :] # sobel in x-direction

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

    image = jnp.pad(image, [[0, 0], [1, 1], [1, 1]])
    output = sorbel_filter(image, kernels)

    return output


def _retrieve_value_at(img, loc, out_of_bound_value=0):

    iloc = jnp.floor(loc).astype(int)
    res = loc - iloc

    offsets = jnp.asarray(
        [[(i >> j) % 2 for j in range(len(loc))] for i in range(2 ** len(loc))]
    )
    ilocs = jnp.swapaxes(iloc + offsets, 0, 1)

    weight = jnp.prod(res * (offsets == 1) + (1 - res) * (offsets == 0), axis=1)

    max_indices = jnp.asarray(img.shape)[: len(loc), None]
    values = jnp.where(
        (ilocs >= 0).all(axis=0) & (ilocs < max_indices).all(axis=0),
        jnp.swapaxes(img[tuple(ilocs)], 0, -1),
        out_of_bound_value,
    )

    value = (values * weight).sum(axis=-1)

    return value


def sub_pixel_samples(
    img: ArrayLike,
    locs: ArrayLike,
    out_of_bound_value: float = 0,
    edge_indexing: bool = False,
) -> Array:
    """Retrieve image values as non-integer locations by interpolation

    Args:
        img: Array of shape [D1,D2,..,Dk, ...]
        locs: Array of shape [d1,d2,..,dn, k]
        out_of_bound_value: optional float constant, defualt 0.
        edge_indexing: if True, the index for the first value in img is 0.5, otherwise 0. Default is False

    Returns:
        values: [d1,d2,..,dn, ...], float
    """

    loc_shape = locs.shape
    img_shape = img.shape
    d_loc = loc_shape[-1]

    if edge_indexing:
        locs = locs - 0.5

    img = img.reshape(img_shape[:d_loc] + (-1,))
    locs = locs.reshape(-1, d_loc)
    op = partial(_retrieve_value_at, out_of_bound_value=out_of_bound_value)

    values = jax.vmap(op, in_axes=(None, 0))(img, locs)
    out_shape = loc_shape[:-1] + img_shape[d_loc:]

    values = values.reshape(out_shape)

    return values


def sub_pixel_crop_and_resize(
    img: ArrayLike, bbox: ArrayLike, output_shape: Shape, out_of_bound_value: float = 0
) -> Array:
    """Retrieve image values of a bbox resize output. Used for ROI-Align
    Args:
        img: Array of shape [H, W, ...]
        bbox: [y0, x0, y1, x1]
        output_shape: [h, w]
        out_of_bound_value: optional float constant, defualt 0.
    Returns:
        values: [h, w, ...], float
    """

    bbox = jnp.asarray(bbox)
    img = jnp.asarray(img)
    y0, x0, y1, x1 = bbox
    h, w = output_shape

    dy = (y1 - y0) / h
    dx = (x1 - x0) / w

    yy, xx = jnp.mgrid[(y0 + dy / 2) : y1 : (dy), (x0 + dx / 2) : x1 : dx]
    return sub_pixel_samples(
        img,
        jnp.stack([yy, xx], axis=-1),
        out_of_bound_value=out_of_bound_value,
        edge_indexing=True,
    )
