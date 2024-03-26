from __future__ import annotations

import flax.linen as nn
import jax
import optax
import pytest

import lacss.train


def mse(batch, prediction):
    labels = batch[1]
    return ((prediction - labels) ** 2).mean()


def test_vmap_strategy():
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    _X = jax.random.uniform(k1, [16, 4, 16])
    _Y = jax.random.uniform(k2, [16, 4, 4])

    def gen(X, Y):
        for x, y in zip(X, Y):
            yield x, y

    def _run():
        g = gen(X=_X, Y=_Y)

        for _ in (it := trainer.train(g)):
            pass
        return it.loss["mse"]

    trainer = lacss.train.Trainer(
        model=nn.Dense(4),
        losses=mse,
        optimizer=optax.adam(0.01),
        seed=key,
        strategy=lacss.train.Eager,
    )

    eager_loss = _run()

    trainer = lacss.train.Trainer(
        model=nn.Dense(4),
        losses=mse,
        optimizer=optax.adam(0.01),
        seed=key,
        strategy=lacss.train.VMapped,
    )

    vmap_loss = _run()

    assert jax.numpy.allclose(eager_loss, vmap_loss)
