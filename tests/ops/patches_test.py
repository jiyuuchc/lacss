import jax
import jax.numpy as jnp

from lacss.ops import iou_patches_and_labels


def test_iou_patch_label():
    #  3, 3  out of bound
    # [3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 4, 4, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 4, 1, 1, 0, 0],
    # [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    key = jax.random.PRNGKey(55)
    p = jax.random.uniform(key, [4, 2, 2]) > 0.25
    p = jax.vmap(jnp.pad, in_axes=(0, None))(p, 3).astype("float32")
    yc, xc = jnp.expand_dims(jnp.mgrid[-4:4, -4:4], 1) + (
        jax.random.uniform(key, [2, 4, 1, 1]) * 8
    ).astype(int)

    label = jnp.asarray(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
        ],
        dtype=float,
    )

    iou = iou_patches_and_labels((p, yc, xc), label)

    assert iou.shape == (4, 3)
    assert jnp.count_nonzero(iou) == 2

    assert jnp.allclose(iou[2, 0], jnp.asarray(0.25))
    assert jnp.allclose(iou[3, 2], jnp.asarray(0.5))
