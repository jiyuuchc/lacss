from __future__ import annotations

import jax
import optax
from ml_collections import ConfigDict

from ..ops import match_and_replace, distance_similarity
from ..typing import ArrayLike
from ..utils import deep_update

jnp = jax.numpy

def _compute_losses(module, gt_locs, mask, decoded, config):
    gamma = config.get("detection_loss_gamma", 2)
    delta = config.get("detection_loss_delta", 4.0)

    pred_locs = decoded['pred_locs']
    ref_locs = decoded['ref_locs']
    pred_logits = decoded['logits']

    if mask is not None:
        mask = mask[tuple(ref_locs.transpose().astype(int))]
    else:
        mask = jnp.array([True])

    indices, regr_mask = _assign_label(module, gt_locs, decoded, config)
    gt_regr = gt_locs[indices]

    cls_loss = optax.sigmoid_focal_loss(
        pred_logits, regr_mask, gamma=gamma,
    ).mean(where=mask)

    regr_loss = optax.l2_loss(
        pred_locs/delta, 
        gt_regr/delta,
    ).mean(axis=-1).sum(where = regr_mask)
    regr_loss /= jnp.count_nonzero(regr_mask) + 1e-6

    return dict(
        lpn_localization_loss=regr_loss,
        lpn_detection_loss=cls_loss,
    )


def _assign_label(module, gt_locs, decoded, config):
    locs = decoded['pred_locs']
    ref_locs = decoded['ref_locs']
    assert locs.ndim == 2 and locs.shape[-1] in (2, 3), f"{locs.shape}"
    assert gt_locs.shape[-1] == locs.shape[-1]

    assert ref_locs.ndim == 2 and ref_locs.shape[-1] in (2, 3), f"{ref_locs.shape}"

    detection_roi = config.get("detection_roi", 8)
    n_labels_max = config.get("n_labels_max", 25)
    n_labels_min = config.get("n_labels_min", 4)
    similarity_score_scaling = config.get("similarity_score_scaling", 4)

    distances = gt_locs - ref_locs[:, None, :] #[M, N, 2/3]
    init_mask = (jnp.abs(distances) < detection_roi).all(axis=-1) # [M. N]

    sm = jnp.where(init_mask, distance_similarity(locs, gt_locs), 0) #[M, N]

    M, N = sm.shape

    sm_topk, idx = jax.lax.approx_max_k(sm.transpose(), n_labels_max) # [N, 25]
    sm_topk *= similarity_score_scaling

    k = jnp.sqrt(jnp.clip(sm_topk, 1e-16, 1)).sum(axis=1)
    k = jnp.clip(k, n_labels_min, n_labels_max)
    idx = jnp.where(
        (jnp.arange(n_labels_max) < k[:, None]) & (gt_locs[:, -1:] >= 0),
        idx, M,
    )

    indices = jnp.zeros(M, int) - 1 
    indices = indices.at[idx].set(jnp.arange(N)[:, None]) # this relies on Jax's oob behavior
    regr_mask = indices >= 0

    return indices, regr_mask


def train_fn(
    module,
    image: ArrayLike,
    gt_locations: ArrayLike,
    video_refs: tuple|None = None,
    image_mask: ArrayLike|None = None,
    config: ConfigDict = ConfigDict(),
) -> dict:
    assert image.ndim in (3, 4) and image.shape[-1] == 3, f"wrong img shape {image.shape}"
    assert gt_locations.ndim == 2 and gt_locations.shape[1] in (2,3), f"wrong location data shape {gt_locations.shape}"

    is_3d = image.ndim == 4

    if image_mask is not None:
        image_mask = jnp.broadcast_to(image_mask, image.shape[:-1])

    x_det, x_seg = module.get_features(image, video_refs, deterministic=is_3d)

    if not is_3d:
        assert x_det.ndim == 3
        outputs = module.detector(x_det, image_mask)
    else:
        assert x_det.ndim == 4
        outputs = module.detector_3d(x_det, image_mask)

    outputs['losses'] = _compute_losses(module, gt_locations, image_mask, outputs['detector'], config)

    if not is_3d and module.segmentor is not None:
        max_proposal_offset = config.get("max_proposal_offset", 12)
        max_training_instances = config.get("max_training_instances", 256)

        match_th = 1 / max_proposal_offset / max_proposal_offset
        seg_locs = match_and_replace(
            gt_locations,  outputs["predictions"]["locations"], match_th
        )
        seg_locs = seg_locs[:max_training_instances]

        segmentor_out = module.segmentor(x_seg, seg_locs)

        outputs = deep_update(outputs, segmentor_out)

    elif is_3d and module.segmentor_3d is not None:
        max_proposal_offset = config.get("max_proposal_offset", 12)
        max_training_instances = config.get("max_training_instances", 32)

        match_th = 1 / max_proposal_offset / max_proposal_offset
        seg_locs = match_and_replace(
            gt_locations,  outputs["predictions"]["locations"], match_th
        )
        seg_locs = seg_locs[:max_training_instances]

        segmentor_out = module.segmentor_3d(x_seg, seg_locs)

        outputs = deep_update(outputs, segmentor_out)
    
    return outputs
