from functools import partial
import jax
import optax
from treex.losses import Loss
from ..ops import sorbel_edges
from .detection import _binary_focal_crossentropy

jnp = jax.numpy


def _get_auxnet_prediction(auxnet_out, instance_edge, category=None):
    if auxnet_out.shape[-1] == 1 or category is None:
        edge_pred = auxnet_out[:,:, 0]
    else:
        edge_pred = auxnet_out[:,:, category]
    return edge_pred, instance_edge

def self_supervised_edge_losses(auxnet_out, instance_edge, category=None):
    '''
    Args:
        auxnet_out: [H,W, n_groups]
        instance_edge: [H,W]
        category: category number [0, n_groups)
    return:
        loss: float
    '''
    edge_pred, instance_edge = _get_auxnet_prediction(auxnet_out, instance_edge, category)

    # edge_pred = jax.lax.stop_gradient(jnp.maximum(edge_pred, instance_edge))
    return optax.l2_loss(edge_pred, instance_edge).mean()

def auxnet_losses(auxnet_out, instance_edge, category=None):
    '''
    Args:
        auxnet_out: [H,W, n_groups]
        instance_edge: [H,W]
        category: category number [0, n_groups)
    return:
        loss: float
    '''
    edge_pred, instance_edge = _get_auxnet_prediction(auxnet_out, instance_edge, category)

    return _binary_focal_crossentropy(edge_pred, instance_edge >= 0.5).mean()

class AuxnetLoss(Loss):
    def call(
        self,
        preds: dict,
        group_num: jnp.ndarray = None,
    ):
        if not 'training_locations' in preds:
            return 0.0

        return jax.vmap(auxnet_losses)(
            auxnet_out = preds['auxnet_out'],
            instance_edge = preds['instance_edge'],
            category = group_num.astype(int),
        )

class InstanceEdgeLoss(Loss):
    def call(
        self,
        preds: dict,
        group_num: jnp.ndarray = None,
    ):
        if not 'training_locations' in preds:
            return 0.0

        if group_num is None:
            return jax.vmap(self_supervised_edge_losses)(
                auxnet_out = preds['auxnet_out'],
                instance_edge = preds['instance_edge'],
            )
        else:
            return jax.vmap(self_supervised_edge_losses)(
                auxnet_out = preds['auxnet_out'],
                instance_edge = preds['instance_edge'],
                category = group_num.astype(int)
            )
