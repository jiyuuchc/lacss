import jax
import jax.numpy as jnp

from lacss.tracking import *


def test_seq_choice():
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
    h, _ = seq_choice(pt, jnp.asarray(0.0), key=ckey, n_sample=4096)
    bins = np.unique(h, axis=0)
    counts = [np.count_nonzero(np.all(h == b, axis=-1)) / len(h) for b in bins]

    assert ((w - jnp.asarray(counts)) ** 2).mean() < 1e-3


def test_seq_choice_order_independence():
    key = jax.random.PRNGKey(1234)
    key, ckey = jax.random.split(key)
    pt = jax.nn.softmax(jax.random.normal(ckey, shape=[3, 3]))

    p_missing = jnp.asarray(0.0)
    key, ckey = jax.random.split(key)
    h1, _ = seq_choice(pt, p_missing, key=ckey, n_sample=40960)
    bins = np.unique(h1, axis=0)
    counts_1 = [np.count_nonzero(np.all(h1 == b, axis=-1)) / len(h1) for b in bins]

    pt = pt[:, ::-1]
    key, ckey = jax.random.split(key)
    h2, _ = seq_choice(pt, p_missing, key=ckey, n_sample=40960)
    h2 = 3 - h2
    bins = np.unique(h2, axis=0)
    counts_2 = [np.count_nonzero(np.all(h2 == b, axis=-1)) / len(h1) for b in bins]

    assert jnp.all((jnp.asarray(counts_1) - jnp.asarray(counts_2)) ** 2 < 1e-4)
