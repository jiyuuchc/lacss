from functools import partial

import flax.linen as nn
import jax
import optax
import pytest

import lacss.train


class Mse(lacss.train.Loss):
    def call(self, preds, target, **kwargs):
        return ((preds - target) ** 2).mean()


def test_vmap_strategy():
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    _X = jax.random.uniform(k1, [16, 4, 16])
    _Y = jax.random.uniform(k2, [16, 4, 4])

    def gen(X, Y):
        for x, y in zip(X, Y):
            yield x, y

    def _run():
        g = partial(gen, X=_X, Y=_Y)

        trainer.initialize(g)

        for log in trainer.train(g):
            pass
        return log["mse"]

    trainer = lacss.train.Trainer(
        model=nn.Dense(4),
        losses=Mse(),
        optimizer=optax.adam(0.01),
        seed=key,
        strategy=lacss.train.strategy.Eager,
    )

    eager_loss = _run()

    trainer = lacss.train.Trainer(
        model=nn.Dense(4),
        losses=Mse(),
        optimizer=optax.adam(0.01),
        seed=key,
        strategy=lacss.train.strategy.VMapped,
    )

    vmap_loss = _run()

    assert jax.numpy.allclose(eager_loss, vmap_loss)
