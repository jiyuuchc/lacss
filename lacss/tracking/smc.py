import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

_to_logits = lambda x: -np.log((1 / x - 1))
_EPS = jnp.finfo("float32").eps


@dataclasses.dataclass(frozen=True)
class HyperParams:
    n_chains: int = 1024
    n_locs: int = 512
    gamma: float = 0.1
    div_avg: float = 30.0
    div_limit: float = 0.1
    div_scale: float = 0.9
    w_miss: float = 0.1
    logit_scale: float = 1.0
    logit_offset: float = 0.0
    miss_weight: float = 1.0
    miss_logit: float = -0.5
    n_sub_sample: int = 256
    p_death: float = 0.02


def _get_tracking_weights(yx0, yx1, gamma):

    delta = (yx1[None, :, :] - yx0[:, None, :]) ** 2
    delta = delta.sum(axis=-1)  # [n_source, n_target]

    wts = jnp.exp(-delta / 2 * gamma * gamma) * gamma * gamma / 0.399

    # remove invalid rows
    wts = jnp.where(yx0[:, :1] >= 0, wts, 0)
    # remove invalid cols
    wts = jnp.where(yx1[:, 0] >= 0, wts, 0)

    return wts


def _compute_div_p(age, hp):

    return (1 - hp.div_limit) / (1 + hp.div_scale ** (age - hp.div_avg))


@partial(jax.jit, static_argnums=3)
@partial(jax.vmap, in_axes=(0, 0, None, None))
def track_to_next_frame(key, cur_frame, yxs, hp):
    """
    Args
        key: rng x n_chains
        cur_frame: dict
            tracked: bool [n_chains, n_loc]
            age: int [n_chains, n_loc]
            parent: int [n_chains, n_loc]
        yxs: a tuple of (yx0, yx1) float [n_loc, 2]
        hp: parameters
    returns:
        next_frame: dict
    """

    yx0, yx1 = yxs
    n_loc = yx0.shape[0]
    n_sample = hp.n_sub_sample

    assert n_loc == cur_frame["tracked"].shape[-1]
    assert n_loc == cur_frame["age"].shape[-1]
    assert n_loc == cur_frame["parent"].shape[-1]
    assert n_loc == yx1.shape[0]

    tracking_weights = _get_tracking_weights(yx0, yx1, hp.gamma)
    tracking_weights += _EPS

    choice_fn = jax.vmap(partial(jax.random.choice, a=n_loc))

    def _vmapped_choice(k, state):
        tracked, parents, log_sample_weight = state

        w = tracking_weights[k] * (1 - tracked)
        log_sample_weight += jnp.log(w.sum(axis=-1))

        ckeys = jax.random.split(jax.random.fold_in(key, k), n_sample)
        choices = choice_fn(ckeys, p=w / w.sum(axis=-1, keepdims=True))
        tracked = tracked.at[jnp.arange(n_sample), choices].set(True)
        parents = parents.at[jnp.arange(n_sample), choices].set(k)

        return tracked, parents, log_sample_weight

    def _for_inner_1(k, state):
        return jax.lax.cond(
            cur_frame["tracked"][k],
            _vmapped_choice,
            lambda _, state: state,
            k,
            state,
        )

    # track to next frame
    log_sample_weight = jnp.zeros([n_sample])
    tracked = jnp.zeros([n_sample, n_loc], dtype=bool)
    parents = jnp.zeros([n_sample, n_loc], dtype=int) - 1

    tracked, parents, log_sample_weight = jax.lax.fori_loop(
        0,
        n_loc,
        _for_inner_1,
        (tracked, parents, log_sample_weight),
    )

    # now handle cell div and add additioal track
    div_p = _compute_div_p(cur_frame["age"], hp)
    key, ckey = jax.random.split(key, 2)
    to_div = jax.random.uniform(ckey, [n_loc]) < div_p

    def _for_inner_2(k, state):
        return jax.lax.cond(
            cur_frame["tracked"][k] & to_div[k],
            _vmapped_choice,
            lambda _, state: state,
            k,
            state,
        )

    tracked, parents, log_sample_weight = jax.lax.fori_loop(
        0,
        n_loc,
        _for_inner_2,
        (tracked, parents, log_sample_weight),
    )

    # select one sample based on sample_weight
    weight = jnp.exp(log_sample_weight - log_sample_weight.max())
    rs = jax.random.choice(key, n_sample, p=weight / weight.sum())

    next_frame = dict(
        tracked=tracked[rs],
        parent=parents[rs],
        age=jnp.where(
            to_div[parents[rs]] | (parents[rs] == -1),
            0,
            cur_frame["age"][parents[rs]] + 1,
        ),
    )

    return next_frame


def sample_first_frame(df, n_cells, hp, frame_no=1):
    n_chains = hp.n_chains
    n_locs = hp.n_locs

    f0 = df.loc[df["frame"] == frame_no]
    scores = jax.nn.softmax(
        _to_logits(f0["score"].to_numpy()[:n_locs]) * hp.logit_scale
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

    age = np.zeros([n_chains, n_locs], dtype=int) + hp.div_avg // 2
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

    cur_frame = cur_chains[-1]
    key, key2 = jax.random.split(key, 2)
    next_frame = track_to_next_frame(
        jax.random.split(key2, n_chains),
        cur_frame,
        (yx0, yx1),
        hp,
    )
    cur_chains.append(next_frame)

    # resample all chains
    logits = _to_logits(f1["score"].to_numpy()[:n_locs]) - hp.logit_offset
    logits *= hp.logit_scale
    if len(logits) < n_locs:
        logits = np.pad(logits, [0, n_locs - len(logits)], constant_values=-100.0)

    target_logit = (logits * next_frame["tracked"]).sum(axis=-1)
    weights = jax.nn.softmax(target_logit)
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
