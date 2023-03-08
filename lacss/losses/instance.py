from functools import partial

import jax

from ..train.loss import Loss

jnp = jax.numpy
EPS = jnp.finfo("float32").eps


def instance_overlap_losses(
    instances, instance_logit, yc, xc, mask, seg, ignore_seg_loss=False
):
    """
    Args:
        instances: [N, H, W]
        instances_logit: [N, H, W]
        yc: [N, H, W]
        xc: [N, H, W]
        mask: [N, 1, 1]
        seg: [img_height, img_width]
        ignore_seg_loss: bool
    return:
        loss:
    """
    n_instances = jnp.count_nonzero(mask)

    patch_size = instances.shape[-1]
    padding_size = patch_size // 2 + 2
    yc += padding_size
    xc += padding_size

    seg = jnp.pad(seg, padding_size).astype(instances.dtype)
    if not ignore_seg_loss:
        loss = (
            (seg.sum() - instances.sum())
            / patch_size
            / patch_size
            / (n_instances + EPS)
        )
        log_yi_sum = (1.0 - seg) * (-2.0)
    else:
        loss = 0.0
        log_yi_sum = jnp.zeros_like(seg)

    log_yi = -jnp.log(1 + jnp.exp(instance_logit))
    log_yi_sum = log_yi_sum.at[yc, xc].add(log_yi)
    log_yi = log_yi_sum[yc, xc] - log_yi

    loss = loss - (instances * log_yi).mean(axis=(1, 2), keepdims=True).sum(
        where=mask
    ) / (n_instances + EPS)

    return loss


def supervised_segmentation_losses(instances, yc, xc, mask, gt_label):
    """
    Args:
        instances: [N, H, W]
        yc: [N, H, W]
        xc: [N, H, W]
        mask: [N, 1, 1]
        gt_label: [img_height, img_width] labels, bg_label=0
    return:
        loss: based on cross entropy
    """
    n_patches, _, _ = yc.shape

    gt_label = gt_label.astype(int)
    gt_label = jnp.pad(gt_label, [[1, 1], [1, 1]])
    gt_patches = gt_label[yc + 1, xc + 1] == (jnp.arange(n_patches)[:, None, None] + 1)

    p_t = gt_patches * instances + (1 - gt_patches) * (1.0 - instances)
    bce = -jnp.log(jnp.clip(p_t, EPS, 1.0))

    loss = bce.mean(axis=(1, 2), keepdims=True).sum(where=mask) / (mask.sum() + EPS)

    return loss


class InstanceLoss(Loss):
    def call(self, binary_mask: jnp.ndarray, preds: dict, **kwargs):
        if not "training_locations" in preds:
            return 0.0
        return jax.vmap(instance_overlap_losses)(
            instances=preds["instance_output"],
            instance_logit=preds["instance_logit"],
            yc=preds["instance_yc"],
            xc=preds["instance_xc"],
            mask=preds["instance_mask"],
            seg=binary_mask,
        )


class InstanceOverlapLoss(Loss):
    def call(self, inputs: dict, preds: dict, **kwargs):
        if not "training_locations" in preds:
            return 0.0
        segs = jnp.ones(inputs["image"].shape[:-1])
        op = partial(instance_overlap_losses, ignore_seg_loss=True)
        return jax.vmap(op)(
            instances=preds["instance_output"],
            instance_logit=preds["instance_logit"],
            yc=preds["instance_yc"],
            xc=preds["instance_xc"],
            mask=preds["instance_mask"],
            seg=segs,
        )


class SupervisedInstanceLoss(Loss):
    def call(self, mask_labels: jnp.ndarray, preds: dict, **kwargs):
        if not "training_locations" in preds:
            return 0.0

        return jax.vmap(supervised_segmentation_losses)(
            instances=preds["instance_output"],
            yc=preds["instance_yc"],
            xc=preds["instance_xc"],
            mask=preds["instance_mask"],
            gt_label=mask_labels,
        )
