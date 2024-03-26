from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest
from flax.training.train_state import TrainState

import lacss.train


def mse(batch, prediction):
    labels = batch[1]
    return ((prediction - labels) ** 2).mean()


def test_checkpoint(tmp_path):
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    _X = jax.random.uniform(k1, [16, 4, 16])
    _Y = jax.random.uniform(k2, [16, 4, 4])

    def gen(X, Y):
        while True:
            for x, y in zip(X, Y):
                yield x, y

    trainer = lacss.train.Trainer(
        model=nn.Dense(4),
        losses=mse,
        optimizer=optax.adam(0.01),
        seed=key,
    )

    g = gen(X=_X, Y=_Y)

    train_it = trainer.train(g)
    for _ in range(8):
        next(train_it)

    checkpointer = ocp.StandardCheckpointer()
    cp_path = tmp_path.absolute()

    checkpointer.save(cp_path / "test_cp", train_it)

    prev_loss = train_it.loss_logs[0].compute()

    for _ in range(8):
        next(train_it)
    new_loss = train_it.loss_logs[0].compute()
    assert new_loss < prev_loss

    restored = checkpointer.restore(
        cp_path / "test_cp",
        train_it,
    )

    assert restored is not None
    assert type(restored) is type(train_it)

    restored_loss = restored.loss_logs[0].compute()

    assert restored_loss == prev_loss

    # tree_is_same = jax.tree_util.tree_map(lambda x, y: jnp.allclose(x,y), train_state, restored)

    # assert jax.tree_util.tree_reduce(lambda x, y: x and y, tree_is_same, True)
