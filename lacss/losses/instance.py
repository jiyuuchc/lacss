import jax
import optax
from treex.losses import Loss
from ..ops import sorbel_edges
from .detection import _binary_focal_crossentropy

jnp = jax.numpy

EPS = jnp.finfo(float).eps

def instance_overlap_losses(pred, yc, xc, binary_mask, n_instances):
    '''
    Args:
        pred: [n_patches, patch_size, patch_size, 1]: float32, prediction 0..1, padded with 0
        yc, xc: [n_patches, patch_size, patch_size]: coordinates, padded with -1
        binary_mask: [img_height, img_width]: int32
        n_instances: int
    '''
    if pred.ndim == 4:
        pred = pred.squeeze(-1)

    patch_size = pred.shape[-1]
    h,w = binary_mask.shape

    padding_size = patch_size//2 + 2
    yc += padding_size
    xc += padding_size
    binary_mask = jnp.pad(binary_mask, padding_size).astype(float)
    log_yi_sum = jnp.zeros_like(binary_mask)
    # log_yi_sum = jnp.zeros([h + padding_size * 2, w + padding_size * 2])

    log_yi = jnp.log(jnp.clip(1.0 - pred, EPS, 1.0))
    log_yi_sum = log_yi_sum.at[yc, xc].add(log_yi)
    log_yi_sum += (1.0 - binary_mask) * (-2.0)
    log_yi = log_yi_sum[yc, xc] - log_yi

    loss = (binary_mask.sum() - pred.sum()) / pred.size - (pred * log_yi).mean()
    # loss = - (pred * log_yi).mean(axis=(1,2))
    
    return loss * pred.shape[0] / (n_instances+1e-8)

def supervised_segmentation_losses(pred, yc, xc, gt_labels, n_instances):
    '''
    Args:
        pred: [n_patches, patch_size, patch_size, -1]: prediction 0..1, padded with 0
        yc, xc: [n_patches, patch_size, patch_size] meshgrid coordinates, padded with -1
        gt_labels: [img_height, img_width] labels, bg_label=0
        n_instances: int
    return:
        loss: based on cross entropy
    '''
    n_patches, _, crop_size = yc.shape
    if pred.ndim == 4:
        pred = pred.squeeze(-1)

    gt_labels = gt_labels.astype(int)
    gt_labels = jnp.pad(gt_labels, [[1,1],[1,1]])
    gt_patches = gt_labels[yc+1, xc+1] == (jnp.arange(n_patches)[:, None, None] + 1)

    p_t = gt_patches * pred + (1 - gt_patches) * (1.0 - pred)
    bce = - jnp.log(jnp.clip(p_t, EPS, 1.0))
    loss = bce.mean(axis=(1,2))
    
    return loss.sum() / n_instances

class InstanceLoss(Loss):
    def call(
        self,
        binary_mask: jnp.ndarray,
        preds: dict, 
        **kwargs,
    ):
        n_instances = jnp.count_nonzero(preds['training_locations'][:, :, 0] >= 0, axis=1)
        return jax.vmap(instance_overlap_losses)(
            preds['instance_output'],
            preds['instance_yc'],
            preds['instance_xc'],
            binary_mask,
            n_instances,
        )

class SupervisedInstanceLoss(Loss):
    def call(
        self,
        mask_labels: jnp.ndarray,
        preds: dict, 
        **kwargs,
    ):
        n_instances = jnp.count_nonzero(preds['training_locations'][:, :, 0] >= 0, axis=1)
        return jax.vmap(supervised_segmentation_losses)(
            preds['instance_output'],
            preds['instance_yc'],
            preds['instance_xc'],
            mask_labels,
            n_instances,
        )

def self_supervised_edge_losses(preds, category):
    '''
    Args:
        preds: Model output
        category: category number [0,C)
    return:
        loss: float
    '''

    auxnet_out = preds['auxnet_out']
    instance_edge = preds['instance_edge']

    if auxnet_out.shape[-1] == 1:
        edge_pred = auxnet_out[:,:, 0]
    else:
        edge_pred = auxnet_out[:,:, category]

    # edge_pred = jax.lax.stop_gradient(jnp.maximum(edge_pred, instance_edge))
    loss = optax.l2_loss(edge_pred, instance_edge).mean()

    return loss

def auxnet_losses(preds, category):
    '''
    Args:
        preds: Model output
        category: category number [0,C)
    return:
        loss: float
    '''
    auxnet_out = preds['auxnet_out']
    instance_edge = preds['instance_edge']

    if auxnet_out.shape[-1] == 1:
        edge_pred = auxnet_out[:,:, 0]
    else:
        edge_pred = auxnet_out[:,:, category]

    loss = _binary_focal_crossentropy(edge_pred, instance_edge >= 0.5).mean()

    return loss

class AuxnetLoss(Loss):
    def call(
        self,
        group_num: jnp.ndarray,
        preds: dict,
        **kwargs
    ):
        return jax.vmap(auxnet_losses)(
            preds,
            group_num.astype(int),
        )

class InstanceEdgeLoss(Loss):
    def call(
        self,
        group_num: jnp.ndarray,
        preds: dict,
        **kwargs
    ):
        return jax.vmap(self_supervised_edge_losses)(
            preds,
            group_num.astype(int),
        )
