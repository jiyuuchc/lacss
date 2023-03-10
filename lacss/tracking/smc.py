import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def sample_unselected(selected, p, p_miss, *, key):
    """
    Args:
        selected: [n_sample, n_target]
        p: [n_target]
        p_miss: [1]
        n: int 1 or 2
        key: rng
    Returns:
        selection: [n_sample, n]
    """

    @jax.vmap
    def _mapped_choice(keys, weights):
        n_target = weights.size
        return jax.random.choice(
            keys,
            n_target,
            shape=[
                1,
            ],
            replace=False,
            p=weights,
        )

    n_sample, n_target = selected.shape

    total_mass = p_miss + p.sum()
    p_miss = p_miss / total_mass
    p = p / total_mass
    w = p * (1 - selected)  # [n_samples, n_targets]
    w = jnp.pad(w, [[0, 0], [0, 1]], constant_values=p_miss)
    w = w / w.sum(axis=-1, keepdims=True)  # noramlze each row
    assert w.shape == (n_sample, n_target + 1)

    keys = jax.random.split(key, n_sample)
    selection = _mapped_choice(keys, w)

    # formatting issue
    selection = jnp.where(selection == n_target, -1, selection)

    return selection


def update_history(history, selected, selection, k):
    n_sample = history.shape[0]
    history = history.at[:, k].set(selection[:, 0])
    # history = jnp.concatenate([history, selection], axis=-1)
    selected = jnp.pad(
        selected, [[0, 0], [0, 1]]
    )  # pad an extra column to catch both -1
    selected = selected.at[jnp.arange(n_sample), selection[:, 0]].set(1)
    selected = selected[:, :-1]  # remove padding

    return history, selected


def sample_and_update_history(history, selected, k, p, p_miss, *, key):
    selection = sample_unselected(selected, p, p_miss, key=key)
    history, selected = update_history(history, selected, selection, k)
    return history, selected


def get_sample_weight(selected, p, p_miss):
    total_mass = p_miss + p.sum()
    p_miss = p_miss / total_mass
    p = p / total_mass

    w = (p * (1 - selected)).sum(axis=-1) + p_miss
    w = w / w.sum()
    return w


def resample_history(history, selected, p, p_miss, *, key):
    n_sample = selected.shape[0]
    w = get_sample_weight(selected, p, p_miss)
    # print(w.shape)
    c = jax.random.choice(key, n_sample, p=w, shape=[n_sample])
    history = history[c]
    selected = selected[c]

    return history, selected


def seq_choice(key, wt, w_miss, w_nd, n_sample, resample=False):
    """
    Args
        key: rng
        wt: ndarray [n_source, n_target]
        w_miss: [n_source]
        w_nd: [n_source]
        n_sample: int
        resample: bool, default False
    returns: a single sample
        history: [n_source]
        selected: [n_target]
    """

    n_source, n_target = wt.shape

    selected = jnp.zeros([n_sample, n_target], dtype=int)

    log_weight = jnp.zeros([n_sample])

    history = jnp.zeros(shape=[n_sample, n_source], dtype=int) - 1
    history_div = jnp.zeros(shape=[n_sample, n_source], dtype=int) - 1

    def _for_inner(k, state):
        history, selected, key, log_weight = state
        w = get_sample_weight(selected, wt[k], w_miss[k])
        if resample:
            key, ckey = jax.random.split(key)
            c = jax.random.choice(ckey, n_sample, p=w, shape=[n_sample])
            history = history[c]
            selected = selected[c]
        else:
            log_weight += jnp.log(w)

        key, ckey = jax.random.split(key)
        history, selected = sample_and_update_history(
            history, selected, k, wt[k], w_miss[k], key=ckey
        )

        return history, selected, key, log_weight

    history, selected, key, log_weight = jax.lax.fori_loop(
        0,
        n_source,
        lambda k, state: jax.lax.cond(
            jnp.any(wt[k] > 0),
            _for_inner,
            lambda _, state: state,
            k,
            state,
        ),
        (history, selected, key, log_weight),
    )

    if not resample:
        weight = jnp.exp(log_weight - log_weight.max())
        weight = weight / weight.sum()
        rs = jax.random.choice(key, n_sample, shape=[n_sample], p=weight)
        selected = selected[rs]
        history = history[rs]
        log_weight = jnp.zeros([n_sample])

    key, ckey = jax.random.split(key)
    div = jax.random.uniform(ckey, w_nd.shape) < 1 / (1 + w_nd)
    wt = jnp.where(div[:, None], wt, 0.0)
    # w_miss += w_nd * (wt.sum(axis=1) + w_miss)
    history_div, selected, key, log_weight = jax.lax.fori_loop(
        0,
        n_source,
        lambda k, state: jax.lax.cond(
            jnp.any(wt[k] > 0),
            _for_inner,
            lambda _, state: state,
            k,
            state,
        ),
        (history_div, selected, key, log_weight),
    )

    if resample:
        rs = 0
    else:
        weight = jnp.exp(log_weight - log_weight.max())
        weight = weight / weight.sum()
        rs = jax.random.choice(key, n_sample, p=weight)

    return history[rs], history_div[rs], selected[rs]


@partial(jax.jit, static_argnums=4)
def _sample_step(key, padded_wts, w_miss, padded_nd, n_sub_sample):
    print(padded_wts.shape)
    n_sample, _, n_target = padded_wts.shape
    k0, k1, k2, k3 = jax.random.split(key, 4)
    k1 = jax.random.split(k1, n_sample)
    k2 = jax.random.split(k2, n_sample)
    k3 = jax.random.split(k3, n_sample)

    # n_target = padded_wts.shape[-1]
    # p_div = 1 / (1 + padded_nd)
    # div = (jax.random.uniform(k0, [n_sample, n_sub_sample, n_target]) <= p_div[:, None, :]).astype(int)
    history, history_div, selected = jax.vmap(
        partial(seq_choice, n_sample=n_sub_sample)
    )(
        k1,
        padded_wts,
        w_miss,
        padded_nd,
    )

    reversed = (history == -1) & (history_div != -1)
    history = jnp.where(reversed, history_div, history)
    history_div = jnp.where(reversed, -1, history_div)

    return history, history_div, selected


def sample_step(key, selected, wts, w_miss, w_nd, n_sub_sample=256):
    """
    Args:
        key: a single key
        selected: [n_sample, n_source]
        wts: [n_source, n_target], pad with 0 will ensure all invalid rows return -1
        w_miss: [n_source]
        w_nd: [n_samples, n_source]
    Returns:
        history: [n_samples, n_source], -1 means missing
        history_div: [n_samples, n_source], -1 means no division
        selected: [n_samples, n_target], 1/0
    """
    ROWPADSIZE = 32
    COLPADSIZE = 32

    n_sample, _ = selected.shape
    n_source, n_target = wts.shape

    wts = np.array(wts)
    w_nd = np.array(w_nd)
    selected = np.asarray(selected).astype(bool)

    actual_n_source = selected.sum(-1)
    padto_source = (actual_n_source.max() - 1) // ROWPADSIZE * ROWPADSIZE + ROWPADSIZE
    padto_target = n_target // COLPADSIZE * COLPADSIZE + COLPADSIZE - 1

    padded_wts = np.stack(
        [
            np.pad(
                wts[selected[k]],
                [[0, padto_source - actual_n_source[k]], [0, padto_target - n_target]],
            )
            for k in range(n_sample)
        ]
    )
    padded_nd = np.stack(
        [
            np.pad(w_nd[k][selected[k]], [0, padto_source - actual_n_source[k]])
            for k in range(n_sample)
        ]
    )
    w_miss = np.stack(
        [
            np.pad(w_miss[selected[k]], [0, padto_source - actual_n_source[k]])
            for k in range(n_sample)
        ]
    )

    padded_history, padded_history_div, padded_selected = _sample_step(
        key, padded_wts, w_miss, padded_nd, n_sub_sample
    )
    padded_history = np.stack([padded_history, padded_history_div], axis=-1)
    history = np.zeros([n_sample, n_source, 2], dtype=int)
    for k in range(n_sample):
        history[k, selected[k], :] = padded_history[k, : actual_n_source[k], :]
    selected = padded_selected[:, :n_target]

    return history, selected


def post_process_step(history):
    selected0 = history["samples"][-2]["selected"]
    selected1 = history["samples"][-1]["selected"]
    link, div = np.moveaxis(history["samples"][-2]["next"], -1, 0)
    lifetime0 = history["samples"][-2]["lifetime"]
    yx1 = history["detections"][-1]["yx_next"]
    yx0 = history["detections"][-1]["yx"]
    id1 = history["detections"][-1]["id_next"]
    id0 = history["detections"][-1]["id"]

    n_target = yx1.shape[0]

    def _update_lifetime(lifetime0, link, div):
        lifetime = jnp.zeros([n_target + 1])
        link_no_div = jnp.where(div != -1, -1, link)
        lifetime = lifetime.at[link_no_div].set(lifetime0 + 1)[:-1]
        lifetime_extra = jnp.where(link == -1, lifetime0 + 1, 0)
        return jnp.concatenate([lifetime, lifetime_extra])

    indicator_for_yx0 = (selected0 == 1) & (link == -1)  # [n_samples, n_source]
    indicator_for_yx1 = selected1 == 1  # [n_samples, n_target]
    indicator_all = np.concatenate(
        [indicator_for_yx1, indicator_for_yx0], axis=-1
    )  # [n_sample, n_source+n+target]
    is_not_empty = indicator_all.any(axis=0)  # [n_source+n_target]

    yx_all = np.concatenate([yx1, yx0])  # [n_sourc+n_target, 2]
    id_all = np.concatenate([id1, id0])
    lifetime1 = np.asarray(jax.vmap(_update_lifetime)(lifetime0, link, div))
    # print(lifetime1.shape)

    history["detections"].append(
        dict(
            yx=yx_all[is_not_empty],
            id=id_all[is_not_empty],
        )
    )
    history["samples"][-1].update(
        dict(
            selected=indicator_all[:, is_not_empty].astype(int),
            lifetime=lifetime1[:, is_not_empty],
        )
    )

    return history


def get_tracking_weights(yx0, yx1, gamma):
    delta = ((yx1[None, :, :] - yx0[:, None, :]) ** 2).sum(
        axis=-1
    )  # [n_source, n_target]
    wts = jnp.exp(-delta / 2 * gamma * gamma) * gamma * gamma / 0.399
    return wts


def compute_div_p(lifetime, avg, scale, limit):
    p0 = (1 / scale) ** avg
    return p0 * (scale**lifetime) + limit


@dataclasses.dataclass
class HyperParams:
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


def track_to_next_frame(key, history, nextframe, hyper_params):
    """
    Args:
        key: rng
        nextframe: dict
          yx: [n_target, 2], yx locations
          id: [n_target] int
          logit: [n_target], logit of each yx1 location
        history: dict['yx': [array,...], 'samples': [{'selected':array, 'history':array, 'history_div':array, 'lifetime':array},...]]
            yx: [n_source, 2], yx locations from frame 0
            selected: [n_samples, n_source]: indicators of source locations that are being tracked
            lifetime: [n_samples, n_source] how long has each sample been without division
        haper_params: HyperParam dataclass
    """
    yx0 = history["detections"][-1]["yx"]
    selected0 = history["samples"][-1]["selected"]
    lifetime = history["samples"][-1]["lifetime"]

    yx1 = np.asarray(nextframe["yx"])
    id1 = np.asarray(nextframe["id"])
    logit = np.asarray(nextframe["logit"])
    target_logit = logit * hyper_params.logit_scale + hyper_params.logit_offset
    miss_logit = (
        hyper_params.miss_logit * hyper_params.logit_scale + hyper_params.logit_offset
    )

    n_sample, _ = selected0.shape
    wts = get_tracking_weights(yx0, yx1, hyper_params.gamma)  # [n_source, n_target]
    w_miss = (1 - wts.sum(axis=1)) * np.exp(miss_logit) * hyper_params.miss_weight

    wts = wts * np.exp(target_logit)

    w_nd = compute_div_p(
        lifetime,
        hyper_params.div_avg,
        hyper_params.div_scale,
        hyper_params.div_limit,
    )

    key, ckey = jax.random.split(key)
    new_history, selected1 = sample_step(
        ckey,
        selected0,
        wts,
        w_miss,
        w_nd,
        n_sub_sample=hyper_params.n_sub_sample,
    )

    # random remove missing cell
    is_missing = selected0 & (new_history[:, :, 0] == -1)
    if hyper_params.p_death > 0:
        key, ckey = jax.random.split(key)
        is_dead = jax.random.uniform(ckey, is_missing.shape) < hyper_params.p_death
        new_history[is_dead & is_missing, 0] = -2

    # save new tracks
    history["detections"][-1].update(
        dict(
            yx_next=yx1,
            id_next=id1,
            logit_next=logit,
        )
    )
    history["samples"][-1].update(
        dict(
            next=new_history,
        )
    )
    history["samples"].append(
        dict(
            selected=np.asarray(selected1),
        )
    )

    # resample all chains
    target_logit = (target_logit * selected1).sum(axis=-1)
    target_logit += is_missing.sum(axis=1) * hyper_params.miss_logit
    weights = jax.nn.softmax(target_logit)
    rs = np.asarray(jax.random.choice(key, n_sample, [n_sample], p=weights))
    history["samples"] = jax.tree_map(lambda v: v[rs], history["samples"])

    history = post_process_step(history)

    return history
