from __future__ import annotations

from functools import partial

import numpy as np
import jax
from jax.scipy.signal import convolve

jnp = jax.numpy

from ..typing import *


def sorbel_edges_3d(images: ArrayLike) -> Array:
    """Returns a tensor holding Sobel edge maps in 3d.

    Args:
        images: [n, d, h, w]

    Returns:
        Tensor holding edge maps for each channel. [3, n, d, h, w]
    """
    kernels_z = jnp.array([
       [[ 0, -1,  0],
        [-1, -2, -1],
        [ 0, -1,  0]],
       [[ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0]],
       [[ 0,  1,  0],
        [ 1,  2,  1],
        [ 0,  1,  0]]
    ])        
    kernels_y = jnp.array([
       [[ 0, -1,  0],
        [ 0,  0,  0],
        [ 0,  1,  0]],
       [[-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]],
       [[ 0, -1,  0],
        [ 0,  0,  0],
        [ 0,  1,  0]]
    ])
    kernel_x = jnp.array([
       [[ 0,  0,  0],
        [-1,  0,  1],
        [ 0,  0,  0]],
       [[-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]],
       [[ 0,  0,  0],
        [-1,  0,  1],
        [ 0,  0,  0]]])
    kernels = jnp.stack([kernels_z, kernels_y, kernel_x])
    sorbel_filter = jax.vmap(
        jax.vmap(
            partial(convolve, mode="valid"),
            in_axes=[0, None],
        ),
        in_axes=[None, 0],
    )

    images = jnp.pad(images, [[0, 0], [1, 1], [1, 1], [1, 1]])
    output = sorbel_filter(images, kernels)

    return output


def sorbel_edges(images: ArrayLike) -> Array:
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

    images = jnp.pad(images, [[0, 0], [1, 1], [1, 1]])
    output = sorbel_filter(images, kernels)

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
        edge_indexing: if True, the index for the top/left pixel is 0.5, otherwise 0. Default is False

    Returns:
        values: [d1,d2,..,dn, ...], float
    """

    loc_shape = locs.shape
    img_shape = img.shape
    d_loc = loc_shape[-1]
    locs = jnp.asarray(locs)
    img = jnp.asarray(img)

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
    img: ArrayLike,
    bbox: ArrayLike,
    output_shape: tuple[int],
    out_of_bound_value: float = 0,
) -> Array:
    """Retrieve image values of a bbox. Resize output to output_shape. Used for ROI-Align.

    Args:
        img: Array of shape [H, W, ...] (2D) or [D, H, W, ...] (3D)
        bbox: [y0, x0, y1, x1] (2D) or [z0, y0, x0, z1, y1, x1] (3D)
        output_shape: [h, w] (2D) or [d, h, w] (3D)
        out_of_bound_value: optional float constant, defualt 0.

    Returns:
        values: [h, w, ...] or [d, h, w, ...]
    """

    bbox = jnp.asarray(bbox).reshape(2, -1)
    img = jnp.asarray(img)
    output_shape = tuple(output_shape)

    assert bbox.shape[1] == len(output_shape), f"bbox dim is {bbox.shape[1]}, but output_shape dim is {len(output_shape)}."
    assert bbox.shape[1] <= img.ndim, f"bbox dim is {bbox.shape[1]}, but img dim {img.ndim} is smaller."

    delta = (bbox[1] - bbox[0]) / np.asarray(output_shape)
    slices = [slice(d) for d in output_shape]
    g = jnp.moveaxis(jnp.mgrid[slices], 0, -1)
    g = g * delta
    g = g + bbox[0] + delta / 2
    
    return sub_pixel_samples(
        img, g,
        out_of_bound_value=out_of_bound_value,
        edge_indexing=True,
    )
