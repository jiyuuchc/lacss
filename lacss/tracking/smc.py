import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

_to_logits = lambda x: -jnp.log((1 / x - 1))
_EPS = jnp.finfo("float32").eps
_MIN = jnp.finfo("float32").min


@dataclasses.dataclass(frozen=True)
class HyperParams:
    n_chains: int = 4096
    n_locs: int = 512
    gamma: float = 0.1
    logit_scale: float = 4.0
    logit_offset: float = 1.0
    div_avg: float = 50.0
    div_limit: float = 0.9
    div_scale: float = 0.9
    death_avg: float = 50.0
    death_limit: float = 0
    death_scale: float = 0.9


def _get_tracking_weights(data, hp):

    yx0, yx1, logits = data
    gamma = hp.gamma

    delta = (yx1[None, :, :] - yx0[:, None, :]) ** 2
    delta = delta.sum(axis=-1)  # [n_source, n_target]

    wts = -delta / 2 * gamma * gamma
    wts = wts + logits * hp.logit_scale + hp.logit_offset

    # remove invalid
    wts = jnp.where(yx0[:, :1] >= 0, wts, _MIN)
    wts = jnp.where(yx1[:, 0] >= 0, wts, _MIN)

    return wts


def _compute_div_p(age, hp):

    return hp.div_limit / (1 + hp.div_scale ** (age - hp.div_avg))


def _compute_death_p(age, hp):

    return hp.death_limit / (1 + hp.death_scale ** (age - hp.death_avg))


@partial(jax.jit, static_argnums=3)
@partial(jax.vmap, in_axes=(0, 0, None, None))
def track_to_next_frame(key, cur_frame, data, hp):
    """
    Args
        key: rng x n_chains
        cur_frame: dict
            tracked: bool [n_chains, n_loc]
            age: int [n_chains, n_loc]
            parent: int [n_chains, n_loc]
        data: a tuple of (yx0, yx1, logits) float
        hp: parameters
    returns:
        next_frame: dict
        sample_weight: float
    """

    n_loc = hp.n_locs

    assert n_loc == cur_frame["tracked"].shape[-1]
    assert n_loc == cur_frame["age"].shape[-1]
    assert n_loc == cur_frame["parent"].shape[-1]

    key, key1, key2 = jax.random.split(key, 3)

    prev_tracked = cur_frame["tracked"]
    tracking_weights = _get_tracking_weights(data, hp)

    # death states
    death_p = _compute_death_p(cur_frame["age"], hp)
    death_p = jnp.where(prev_tracked, death_p, 1)
    to_track = jax.random.uniform(key1, [n_loc]) >= death_p

    # generate cell div states
    shuffled = jax.random.permutation(key, n_loc)
    div_p = _compute_div_p(cur_frame["age"], hp)
    div_p = jnp.where(prev_tracked, div_p, 0)
    to_div = jax.random.uniform(key2, [n_loc]) < div_p

    tracked = jnp.zeros([n_loc], dtype=bool)
    parents = jnp.zeros([n_loc], dtype=int) - 1

    def _get_next(k, tracked, parents):

        # use -inf to ensure no duplicate selections
        w = jnp.where(tracked, -jnp.inf, tracking_weights[k])
        choice = jnp.argmax(w)

        tracked = tracked.at[choice].set(True)
        parents = parents.at[choice].set(k)

        return tracked, parents

    def _scan_inner_1(k, state):

        tracked, parents = state
        k = shuffled[k]

        return jax.lax.cond(
            to_track[k],
            _get_next,
            lambda _, tracked, parents: (tracked, parents),
            k,
            tracked,
            parents,
        )

    def _scan_inner_2(k, state):

        tracked, parents = state
        k = shuffled[k]

        return jax.lax.cond(
            to_div[k],
            _get_next,
            lambda _, tracked, parents: (tracked, parents),
            k,
            tracked,
            parents,
        )

    tracked, parents = jax.lax.fori_loop(
        0,
        n_loc,
        _scan_inner_1,
        (tracked, parents),
    )
    tracked, parents = jax.lax.fori_loop(
        0,
        n_loc,
        _scan_inner_2,
        (tracked, parents),
    )
    age = jnp.where(
        to_div[parents],
        0,
        cur_frame["age"][parents] + 1,
    )
    age = jnp.where(tracked, age, -1)

    next_frame = dict(tracked=tracked, parent=parents, age=age)

    sample_weight = tracking_weights[parents, np.arange(n_loc)]
    sample_weight = sample_weight.sum(where=tracked)

    return next_frame, sample_weight


def sample_first_frame(df, n_cells, hp, frame_no=1):
    n_chains = hp.n_chains
    n_locs = hp.n_locs

    f0 = df.loc[df["frame"] == frame_no]
    scores = jax.nn.softmax(
        # _to_logits(f0["score"].to_numpy()[:n_locs]) * hp.logit_scale
        _to_logits(f0["score"].to_numpy()[:n_locs])
        * 10
    )
    scores = np.array(scores)
    yx0 = f0[["y", "x"]].to_numpy()[:n_locs]

    if len(yx0) < n_locs:
        yx0 = np.pad(yx0, [[0, n_locs - len(yx0)], [0, 0]], constant_values=-1)

    tracked = np.zeros([n_chains, n_locs], dtype=bool)
    for k in range(n_chains):
        sel = np.random.choice(
            len(scores), [n_cells], p=scores / scores.sum(), replace=False
        )
        tracked[k, sel] = True

    age = np.random.randint(0, hp.div_avg, size=[n_chains, n_locs])
    parent = np.zeros([n_chains, n_locs], dtype=int) - 1
    cur_frame = dict(
        tracked=jnp.asarray(tracked),
        parent=jnp.asarray(parent),
        age=jnp.asarray(age),
    )

    return cur_frame, yx0


def extend_chains(key, df, cur_chains, yx0, frame_no, hp):
    n_chains = hp.n_chains
    n_locs = hp.n_locs

    f1 = df.loc[df["frame"] == frame_no]
    yx1 = f1[["y", "x"]].to_numpy()[:n_locs]

    if len(yx1) < n_locs:
        yx1 = np.pad(yx1, [[0, n_locs - len(yx1)], [0, 0]], constant_values=-1)

    logits = _to_logits(f1["score"].to_numpy()[:n_locs])
    if len(logits) < n_locs:
        logits = np.pad(logits, [0, n_locs - len(logits)], constant_values=_MIN)

    cur_frame = cur_chains[-1]
    key, key2 = jax.random.split(key, 2)
    next_frame, weights = track_to_next_frame(
        jax.random.split(key2, n_chains),
        cur_frame,
        (yx0, yx1, logits),
        hp,
    )
    cur_chains.append(next_frame)

    # resample all chains

    weights = jax.nn.softmax(weights)
    rs = np.asarray(jax.random.choice(key, len(weights), [n_chains], p=weights))
    cur_chains = jax.tree_map(lambda v: v[rs], cur_chains)

    return cur_chains, yx1


def do_tracking(key, df, hp, starting_n_cell=1, first_frame=None, last_frame=None):

    if first_frame is None:
        first_frame = df["frame"].min()
    if last_frame is None:
        last_frame = df["frame"].max()

    frame, yx = sample_first_frame(df, starting_n_cell, hp, frame_no=first_frame)

    chain = [frame]
    yxs = [yx]
    for frame_no in tqdm(range(first_frame + 1, last_frame + 1)):
        chain, yx = extend_chains(
            jax.random.fold_in(key, frame_no), df, chain, yx, frame_no, hp
        )
        yxs.append(yx)

    # transpose the output and convert to np array
    chain = jax.tree_util.tree_map(lambda *v: np.stack(v), *chain)
    yxs = np.stack(yxs)

    return chain, yxs
