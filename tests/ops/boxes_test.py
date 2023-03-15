import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lacss.ops import box_iou_similarity


def test_box_iou_similarity_batched():
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    boxes_1 = jax.random.uniform(key1, [32, 256, 4])
    boxes_2 = jax.random.uniform(key1, [32, 128, 4])
    ious = box_iou_similarity(boxes_1, boxes_2)

    assert ious.shape == (32, 256, 128)
