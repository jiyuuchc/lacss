import jax
import jax.numpy as jnp
import numpy as np
import dataclasses
from functools import partial

def sample_unselected(selected, p, p_miss, *, key):
    @jax.vmap
    def _mapped_choice(keys, weights):
        n_target = weights.size
        return jax.random.choice(keys, n_target, shape=[1,], p = weights)

    n_sample, n_target = selected.shape
    total_mass = p_miss + p.sum()
    p_miss = p_miss / total_mass
    p = p / total_mass
    w =  p * (1 - selected) # [n_samples, n_targets]
    w = jnp.pad(w, [[0,0],[0,1]], constant_values=p_miss)
    w = w / w.sum(axis=-1, keepdims=True) # noramlze each row
    assert w.shape == (n_sample, n_target+1)

    keys = jax.random.split(key, n_sample)
    selection = _mapped_choice(keys, w)
    
    # formatting issue
    selection = jnp.where(selection == n_target, -1, selection)  

    return selection

def update_history(history, selected, selection, k):
    n_sample = history.shape[0]
    history = history.at[:,k].set(selection[:,0])
    # history = jnp.concatenate([history, selection], axis=-1)
    selected = jnp.pad(selected, [[0,0],[0,1]]) # pad an extra column to catch both -1
    selected = selected.at[jnp.arange(n_sample), selection[:,0]].set(1)
    selected = selected[:, :-1] #remove padding

    return history, selected

def weight_history(selected, p, p_miss):
    total_mass = p_miss + p.sum()
    p_miss = p_miss / total_mass
    p = p / total_mass

    w = (p * (1 - selected)).sum(axis=-1) + p_miss
    return w

def resample_history(history, selected, p, p_miss, *, key):
    n_sample = selected.shape[0]
    w = weight_history(selected, p, p_miss)
    w = w / w.sum()
    # print(w.shape)
    c = jax.random.choice(key, n_sample, p=w, shape=[n_sample])
    history = history[c]
    selected = selected[c]

    return history, selected

def seq_choice(wt, w_miss, prev=None, * , key, n_sample=256, resampling=True):
    '''
    Args
        wt: ndarray [M x N]
        w_miss: zero or 1d array, if 1d size is [M], 
        n_samples: int optional
        prev: when sampling division, prev is not None, and w_miss represent weight for no-division
    returns: a single sample
        history: [n_sample, M]
        selected: [n_sample, N]
    '''

    w_miss = jnp.asarray(w_miss)
    w_miss = w_miss.reshape(w_miss.size)

    n_source, n_target = wt.shape
    log_weight = jnp.zeros([n_sample])

    if prev is None:
        history = jnp.zeros(shape=[n_sample, n_source], dtype=int)
        selected = jnp.zeros(shape=[n_sample, n_target], dtype=int)
    elif resampling: # for sampling cell div, a selection is provided
        raise ValueError('cannot resample if supplied with previous results')
    else:
        selected = prev
        history = jnp.zeros(shape=[n_sample, n_source], dtype=int)
        log_weight = jnp.log(weight_history(selected, wt[0], w_miss[0]))

    def _for_inner(k, state):
        history, selected, key, log_weight = state
        key, sample_key, resample_key= jax.random.split(key, 3)

        # sample current availabel position
        selection = sample_unselected(selected, wt[k], w_miss[k], key=sample_key)

        # update history
        history, selected = update_history(history, selected, selection, k)

        # resample based on next row
        if resampling:
            history, selected = resample_history(history, selected, wt[k+1], w_miss[k+1], key=resample_key)
        else:
            log_weight += jnp.log(weight_history(selected, wt[k+1], w_miss[k+1]))

        return history, selected, key, log_weight
    
    history, selected, key, log_weight = jax.lax.fori_loop(
        0, n_source-1,
        _for_inner,
        (history, selected, key, log_weight),
    )

    # final iter
    k = n_source -1 
    selection = sample_unselected(selected, wt[k], w_miss[k], key=key)
    history, selected = update_history(history, selected, selection, k)

    if resampling:
        return history, selected
    else:
        return history, selected, log_weight

def sample_step(key, wts, w_miss, w_nd, n_sub_sample=256):
    '''
    Args:
        key: a single key
        wts: [n_samples, n_source, n_target], pad with 0 will ensure all invalid rows return -1
        w_miss: float constant
        w_nd: [n_samples, n_source]
        target_logit: [n_target]
    Returns:
        history: [n_samples, n_source], -1 means missing
        history_div: [n_samples, n_source], -1 means no division
        selected: [n_samples, n_target], 1/0
    '''
    n_sample, n_source, n_target = wts.shape
    key, k1, k2, k3 = jax.random.split(key, 4)
    k1 = jax.random.split(k1, n_sample)
    k2 = jax.random.split(k2, n_sample)
    k3 = jax.random.split(k3, n_sample)

    w_miss = jnp.asarray(w_miss).repeat(n_sample).reshape([n_sample,1])
    history, selected = jax.vmap(seq_choice)(wts, w_miss, key=k1)
    history_div, selected, weights = jax.vmap(partial(seq_choice, resampling=False))(wts, w_nd, selected, key=k2)
    weights = jnp.exp(weights)
    weights /= weights.sum(axis=-1, keepdims=True)
    rs = jax.vmap(partial(jax.random.choice, a = weights.shape[1]))(k3, p=weights)

    history = history[jnp.arange(n_sample), rs] #just need one sample
    history_div = history_div[jnp.arange(n_sample), rs]
    selected = selected[jnp.arange(n_sample), rs]

    reversed = (history == -1) & (history_div != -1)
    tmp_history = jnp.where(reversed, history_div, history)
    history_div = jnp.where(reversed, history, history_div)
    history = tmp_history

    return history, history_div, selected

def post_process_step(history):
    selected0 = history['samples'][-2]['selected']
    selected1 = history['samples'][-1]['selected']
    link = history['samples'][-2]['history']
    div = history['samples'][-2]['history_div']
    lifetime0 = history['samples'][-2]['lifetime']
    yx1 = history['detections'][-1]['yx']
    yx0 = history['detections'][-2]['yx']
    id1 = history['detections'][-1]['id']
    id0 = history['detections'][-2]['id']

    n_target = yx1.shape[0]

    @jax.vmap
    def _update_lifetime(lifetime0, link, div):
        lifetime = jnp.zeros([n_target+1])
        link_no_div = jnp.where(div != -1, -1, link)
        lifetime = lifetime.at[link_no_div].set(lifetime0 + 1)[:-1]
        lifetime_extra = jnp.where(link == -1, lifetime0 + 1, 0)
        return jnp.concatenate([lifetime, lifetime_extra])

    indicator_for_yx0 = (selected0 == 1) & (link == -1) # [n_samples, n_source]
    indicator_for_yx1 = (selected1 == 1) # [n_samples, n_target]
    indicator_all = np.concatenate([indicator_for_yx1, indicator_for_yx0], axis=-1) #[n_sample, n_source+n+target]
    is_not_empty = indicator_all.any(axis=0) #[n_source+n_target]

    yx_all = np.concatenate([yx1, yx0]) #[n_sourc+n_target, 2]
    id_all = np.concatenate([id1, id0])
    lifetime1 = np.asarray(_update_lifetime(lifetime0, link, div))
    # print(lifetime1.shape)

    history['detections'][-1].update(dict(
        yx = yx_all[is_not_empty],
        id = id_all[is_not_empty],
    ))
    history['samples'][-1].update(dict(
        selected = indicator_all[:, is_not_empty],
        lifetime = lifetime1[:, is_not_empty],
    ))

    return history

def get_tracking_weights(yx0, yx1, gamma):
    delta = ((yx1[None,:,:] - yx0[:, None,:]) ** 2).sum(axis=-1) #[n_source, n_target]
    wts = jnp.exp(-delta * gamma * gamma)
    return wts

def compute_div_p(lifetime, start, scale, limit):
    return jnp.exp((jnp.log(scale) * lifetime)) * (start-limit) + limit

@dataclasses.dataclass
class HyperParams:
    gamma:float = 0.1
    div_start_w: float = 50
    div_limit: float = 0.5
    div_scale: float = 0.3
    w_miss: float = 0.02
    logit_scale:float = 1.
    logit_offset: float = 0.
    n_sub_sample: int = 64

def track_to_next_frame(key, history, nextframe, hyper_params):
    '''
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
    '''
    yx0 = history['detections'][-1]['yx']
    selected0 = history['samples'][-1]['selected']
    lifetime = history['samples'][-1]['lifetime']
    n_sample, _ = selected0.shape

    yx1 = np.asarray(nextframe['yx'])
    id1 = np.asarray(nextframe['id'])
    logit = np.asarray(nextframe['logit'])

    wts = get_tracking_weights(yx0, yx1, hyper_params.gamma) # [n_source, n_target]
    wts = jnp.where(selected0[:,:,None], wts[None,:,:,], 0.) #masking out unselected source locations

    w_nd = compute_div_p(lifetime, hyper_params.div_start_w, hyper_params.div_scale, hyper_params.div_limit)
    new_history, new_history_div, selected1 = sample_step(key, wts, hyper_params.w_miss, w_nd, n_sub_sample=hyper_params.n_sub_sample)

    history['detections'].append(dict(
        yx = yx1,
        id = id1,
        logit = logit,
    ))
    history['samples'][-1].update(dict(
        history = np.asarray(new_history),
        history_div = np.asarray(new_history_div),
    ))
    history['samples'].append(dict(
        selected = np.asarray(selected1),
    ))

    target_logit = logit * hyper_params.logit_scale + hyper_params.logit_offset
    weights = jax.nn.softmax((target_logit * selected1).sum(axis=-1))
    rs = np.asarray(jax.random.choice(key, n_sample, [n_sample], p = weights))
    
    history['samples'] = jax.tree_map(lambda v:v[rs], history['samples'])

    history = post_process_step(history)

    return history
