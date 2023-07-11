import jax
import jax.numpy as jnp
import pytest

# from lacss.tracking import *


@pytest.mark.skip
def test_seq_choice_with_resampling():
    n_sample = 4096

    key = jax.random.PRNGKey(1234)
    key, ckey = jax.random.split(key)
    pt = jax.nn.softmax(jax.random.normal(ckey, shape=[3, 3]))
    seq = (
        jnp.array(((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)))
        - 1
    )
    w = jax.vmap(lambda x: pt[jnp.arange(3), x].prod())(seq)
    w = w / w.sum()

    key, ckey = jax.random.split(key)
    selected = jnp.zeros([n_sample, 3], dtype=int)
    h, s = seq_choice_with_resampling(ckey, pt, jnp.asarray(0.0), selected)

    assert h.shape == (n_sample, 3)
    assert s.shape == (n_sample, 3)
    assert s.sum() == n_sample * 3


@pytest.mark.skip
def test_seq_choice_with_resampling_numerial_accuracy():
    n_sample = 4096

    key = jax.random.PRNGKey(1234)
    key, ckey = jax.random.split(key)
    pt = jax.nn.softmax(jax.random.normal(ckey, shape=[3, 3]))
    seq = (
        jnp.array(((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)))
        - 1
    )
    w = jax.vmap(lambda x: pt[jnp.arange(3), x].prod())(seq)
    w = w / w.sum()

    key, ckey = jax.random.split(key)
    selected = jnp.zeros([n_sample, 3], dtype=int)
    h, _ = seq_choice_with_resampling(ckey, pt, jnp.asarray(0.0), selected)
    bins = np.unique(h, axis=0)
    counts = [np.count_nonzero(np.all(h == b, axis=-1)) / len(h) for b in bins]

    assert ((w - jnp.asarray(counts)) ** 2).mean() < 1e-3


@pytest.mark.skip
def test_seq_choice_with_resampling_order_independence():
    n_sample = 40960

    key = jax.random.PRNGKey(1234)
    key, ckey = jax.random.split(key)
    pt = jax.nn.softmax(jax.random.normal(ckey, shape=[3, 3]))

    p_missing = jnp.asarray(0.0)
    key, ckey = jax.random.split(key)
    selected = jnp.zeros([n_sample, 3], dtype=int)
    h1, _ = seq_choice_with_resampling(ckey, pt, p_missing, selected)
    bins = np.unique(h1, axis=0)
    counts_1 = [np.count_nonzero(np.all(h1 == b, axis=-1)) / len(h1) for b in bins]

    pt = pt[:, ::-1]
    key, ckey = jax.random.split(key)
    h2, _ = seq_choice_with_resampling(ckey, pt, p_missing, selected)
    h2 = 3 - h2
    bins = np.unique(h2, axis=0)
    counts_2 = [np.count_nonzero(np.all(h2 == b, axis=-1)) / len(h1) for b in bins]

    assert jnp.all((jnp.asarray(counts_1) - jnp.asarray(counts_2)) ** 2 < 1e-4)
