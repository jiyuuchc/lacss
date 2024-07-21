from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lacss.ops import sub_pixel_crop_and_resize

def test_sub_pixel_crop_and_resize():
    # 2d 
    img = np.arange(100).reshape(10, 10, 1)
    box = np.array([1,1,3,3])
    output_shape = (5, 5)

    out = sub_pixel_crop_and_resize(img, box, output_shape)

    assert out.shape == output_shape + (1,)

    # 3d 
    img = np.arange(1000).reshape(10, 10, 10, 1)
    box = np.array([1,2,3,6,5,4])
    output_shape = (1, 4, 4)

    out = sub_pixel_crop_and_resize(img, box, output_shape)

    assert out.shape == output_shape + (1,)

    # vmapped
    img = np.arange(1000).reshape(10, 10, 10, 1)
    box = np.array([1,2,3,6,5,4])
    output_shape = (1, 4, 4)

    img = np.repeat(img[None,...], 8, axis=0)    
    box = np.repeat(box[None,...], 8, axis=0)

    out = jax.vmap(
        partial(sub_pixel_crop_and_resize, output_shape=output_shape)
    )(img, box)

    assert out.shape == (8,) + output_shape + (1,)
    
