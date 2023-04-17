from functools import partial

import jax
import optax

from ..train.loss import Loss

jnp = jax.numpy
EPS = jnp.finfo("float32").eps


def instance_overlap_losses(
    instances,
    instance_logit,
    yc,
    xc,
    mask,
    seg,
    ignore_seg_loss=False,
    phi=0.66,
    alpha=1,
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
    n_instances = jnp.count_nonzero(mask) + EPS

    patch_size = instances.shape[-1]
    padding_size = patch_size // 2 + 2
    yc += padding_size
    xc += padding_size

    seg = jnp.pad(seg, padding_size).astype(instances.dtype)
    log_yi_sum = jnp.zeros_like(seg)
    if not ignore_seg_loss:
        seg_patch = seg[yc, xc]
        loss = (1.0 - seg_patch) * instances + seg_patch * (1.0 - instances)
    else:
        loss = jnp.zeros_like(instances)

    # max_logit = int(jnp.finfo(instance_logit.dtype).maxexp * 0.6932) - 1
    # instance_logit = jnp.where(instance_logit > max_logit, max_logit, instance_logit)

    # log_yi = - jax.nn.log_sigmoid(-instance_logit)
    log_yi = instances
    log_yi_sum = log_yi_sum.at[yc, xc].add(log_yi)
    log_yi = log_yi_sum[yc, xc] - log_yi
    # log_yi /= alpha
    log_yi = (log_yi / phi) ** alpha

    loss = loss + (instances * log_yi)
    loss = loss.mean(axis=(1, 2), keepdims=True).sum(where=mask) / n_instances

    return loss


def supervised_segmentation_losses(logit, yc, xc, mask, gt_label):
    """
    Args:
        logit: [N, H, W]
        yc: [N, H, W]
        xc: [N, H, W]
        mask: [N, 1, 1]
        gt_label: [img_height, img_width] labels, bg_label=0
    return:
        loss: based on cross entropy
    """
    n_patches, ps, _ = yc.shape
    n_instances = jnp.count_nonzero(mask) + EPS

    gt_label = gt_label.astype(int)
    gt_label = jnp.pad(gt_label, ps // 2)
    gt_patches = gt_label[yc + ps // 2, xc + ps // 2] == (
        jnp.arange(n_patches)[:, None, None] + 1
    )
    gt_patches = gt_patches.astype(int)

    loss = optax.sigmoid_binary_cross_entropy(logit, gt_patches)
    loss = loss.mean(axis=(1, 2), keepdims=True).sum(where=mask) / n_instances

    # p_t = gt_patches * instances + (1 - gt_patches) * (1.0 - instances)
    # bce = -jnp.log(jnp.clip(p_t, EPS, 1.0))

    # loss = bce.mean(axis=(1, 2), keepdims=True).sum(where=mask) / (mask.sum() + EPS)

    return loss


class InstanceLoss(Loss):
    def __init__(self, phi=0.7, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.phi = phi
        self.alpha = alpha

    def call(self, preds: dict, labels: dict, **kwargs):
        if not "training_locations" in preds:
            return 0.0
        return instance_overlap_losses(
            instances=preds["instance_output"],
            instance_logit=preds["instance_logit"],
            yc=preds["instance_yc"],
            xc=preds["instance_xc"],
            mask=preds["instance_mask"],
            seg=labels["binary_mask"],
            phi=self.phi,
            alpha=self.alpha,
        )


class InstanceOverlapLoss(Loss):
    def __init__(self, phi=0.7, alpha=1, **kwargs):
        super().__init__(**kwargs)

        self.phi = phi
        self.alpha = alpha

    def call(self, inputs: dict, preds: dict, **kwargs):
        if not "training_locations" in preds:
            return 0.0

        segs = jnp.ones(inputs["image"].shape[:-1])
        return instance_overlap_losses(
            instances=preds["instance_output"],
            instance_logit=preds["instance_logit"],
            yc=preds["instance_yc"],
            xc=preds["instance_xc"],
            mask=preds["instance_mask"],
            seg=segs,
            ignore_seg_loss=True,
            phi=self.phi,
            alpha=self.alpha,
        )


def _mean_over_boolean_mask(loss, mask):

    mask = mask.reshape(mask.shape[0], 1)
    n_instances = jnp.count_nonzero(mask) + EPS

    loss = loss.reshape(loss.shape[0], -1)
    loss = loss.mean(axis=1, keepdims=True).sum(where=mask)
    loss /= n_instances

    return loss


class SupervisedInstanceLoss(Loss):
    def call(self, preds: dict, labels: dict, **kwargs):

        instance_mask = preds["instance_mask"]
        instance_logit = preds["instance_logit"]
        yc = preds["instance_yc"]
        xc = preds["instance_xc"]

        if isinstance(labels, dict):
            gt_labels = labels["gt_labels"].astype("int32")
        else:
            gt_labels = labels.astype("int32")

        n_patches, ps, _ = yc.shape

        gt_labels = jnp.pad(gt_labels, ps // 2)
        gt_patches = gt_labels[yc + ps // 2, xc + ps // 2] == (
            jnp.arange(n_patches)[:, None, None] + 1
        )
        gt_patches = gt_patches.astype(int)

        loss = optax.sigmoid_binary_cross_entropy(instance_logit, gt_patches)

        return _mean_over_boolean_mask(loss, instance_mask)


class SelfSupervisedInstanceLoss(Loss):
    def __init__(self, soft_label: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_label = soft_label

    def call(self, preds: dict, **kwargs):

        instance_mask = preds["instance_mask"]
        instances = preds["instance_output"]
        instance_logit = preds["instance_logit"]
        yc = preds["instance_yc"]
        xc = preds["instance_xc"]

        patch_size = instances.shape[-1]
        padding_size = patch_size // 2 + 2
        yc += padding_size
        xc += padding_size

        binary_mask = jax.lax.stop_gradient(jax.nn.sigmoid(preds["fg_pred"]))
        # binary_mask = jax.lax.stop_gradient(preds["fg_pred"])
        seg = jnp.pad(binary_mask, padding_size)

        if self.soft_label:

            seg_patch = seg[yc, xc]

            loss = (1.0 - seg_patch) * instances + seg_patch * (1.0 - instances)

            instance_sum = jnp.zeros_like(seg)
            instance_sum = instance_sum.at[yc, xc].add(instances)
            instance_sum_i = instance_sum[yc, xc] - instances

            loss = loss + instances * instance_sum_i

        else:

            seg = (seg > 0.5).astype(instances.dtype)
            seg_patch = seg[yc, xc]

            loss = (1.0 - seg_patch) * instances + seg_patch * (1.0 - instances)

            log_yi_sum = jnp.zeros_like(seg)
            log_yi = -jax.nn.log_sigmoid(-instance_logit)
            log_yi_sum = log_yi_sum.at[yc, xc].add(log_yi)
            log_yi = log_yi_sum[yc, xc] - log_yi

            loss = loss + (instances * log_yi)

        return _mean_over_boolean_mask(loss, instance_mask)


class WeaklySupervisedInstanceLoss(Loss):
    def __init__(self, ignore_mask: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_mask = ignore_mask

    def call(self, *, preds, labels, inputs):

        instance_mask = preds["instance_mask"]
        instances = preds["instance_output"]
        instance_logit = preds["instance_logit"]
        yc = preds["instance_yc"]
        xc = preds["instance_xc"]

        patch_size = instances.shape[-1]
        padding_size = patch_size // 2 + 2
        yc += padding_size
        xc += padding_size

        if self.ignore_mask:
            seg = jnp.zeros(inputs["image"].shape[:-1])
            seg = jnp.pad(seg, padding_size)
            loss = jnp.zeros_like(instances)
        else:
            if isinstance(labels, dict):
                seg = labels["gt_mask"].astype("float32")
            else:
                seg = labels.astype("float32")
            seg = jnp.pad(seg, padding_size)
            seg_patch = seg[yc, xc]
            loss = (1.0 - seg_patch) * instances + seg_patch * (1.0 - instances)

        log_yi_sum = jnp.zeros_like(seg)

        log_yi = -jax.nn.log_sigmoid(-instance_logit)
        log_yi_sum = log_yi_sum.at[yc, xc].add(log_yi)
        log_yi = log_yi_sum[yc, xc] - log_yi

        loss = loss + (instances * log_yi)

        # instance_sum = jnp.zeros_like(seg)
        # instance_sum = instance_sum.at[yc, xc].add(instances)
        # instance_sum_i = instance_sum[yc, xc] - instances

        # loss = loss + (instances * instance_sum_i)

        return _mean_over_boolean_mask(loss, instance_mask)
