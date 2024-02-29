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
    cpm = ocp.CheckpointManager(tmp_path)

    for _ in range(16):
        _ = next(train_it)

    result = train_it.send(("checkpoint", cpm))

    assert result == "checkpoint - 1"

    train_state = train_it.send(("train_state",))

    assert train_state is not None
    assert type(train_state) is TrainState
    
    restored = cpm.restore(
        cpm.latest_step(),
        args=ocp.args.StandardRestore(train_state),
    )

    assert restored is not None
    assert type(restored) is TrainState

    assert restored is not train_state

    tree_is_same = jax.tree_util.tree_map(lambda x, y: jnp.allclose(x,y), train_state, restored)

    assert jax.tree_util.tree_reduce(lambda x, y: x and y, tree_is_same, True)
