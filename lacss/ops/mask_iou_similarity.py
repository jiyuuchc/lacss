import tensorflow as tf
from .iou_similarity import *

def mask_iou_similarity(gt, pred, patch_size=96):
    ''' Compute mask_iou matrix.
        The method is to compute box iou matrix first and update only non-zero
        entries with the accurate mask_iou

        Args:
            gt: a tuple of (mask_indices, gt_bboxes)
            pred: a tuple of (pred_mask_indices, pred_bboxes)
            patch_size: int constant

            *_mask_indices: RaggedTensor of [n, None, 2] (y,x) indices of all segmented pixels
            *_bboxes: Tensor of [n, 4] in y0x0y1x1 format
        Outputs:
            similarity_matrix: Tensor of [n_pred_instances, n_gt_instances]
    '''
    box_iou = IouSimilarity()

    gt_mask_indices, gt_bboxes = gt
    pred_mask_indices, pred_bboxes = pred

    sm=box_iou(pred_bboxes, gt_bboxes)
    mask_ids = tf.where(sm > 0)
    gt_masks = tf.gather(gt_mask_indices, mask_ids[:,1])
    pred_masks = tf.gather(pred_mask_indices, mask_ids[:,0])
    offsets = tf.gather(pred_bboxes[:,:2], mask_ids[:,0])[:, None, :]

    gt_masks -= offsets
    pred_masks -= offsets

    area = tf.cast(gt_masks.row_lengths() + pred_masks.row_lengths(), tf.int32)
    n_pairs = gt_masks.nrows()

    pred_masks = tf.concat([pred_masks.value_rowids()[:,None], pred_masks.values], axis=-1)
    valid_rows = tf.logical_and(
        tf.logical_and(pred_masks[:,1]>=0, pred_masks[:,1]<patch_size),
        tf.logical_and(pred_masks[:,2]>=0, pred_masks[:,2]<patch_size),
        )
    pred_masks = tf.boolean_mask(pred_masks, valid_rows)
    n_pred_pixels = tf.shape(pred_masks)[0]
    pred_masks = tf.scatter_nd(pred_masks, tf.ones([n_pred_pixels], tf.uint8), (n_pairs, patch_size, patch_size))

    gt_masks =  tf.concat([gt_masks.value_rowids()[:,None], gt_masks.values], axis=-1)
    valid_rows = tf.logical_and(
        tf.logical_and(gt_masks[:,1]>=0, gt_masks[:,1]<patch_size),
        tf.logical_and(gt_masks[:,2]>=0, gt_masks[:,2]<patch_size),
        )
    gt_masks = tf.boolean_mask(gt_masks, valid_rows)
    n_gt_pixels = tf.shape(gt_masks)[0]
    gt_masks = tf.scatter_nd(gt_masks, tf.ones([n_gt_pixels], tf.uint8), (n_pairs, patch_size, patch_size))

    intersects = tf.reduce_sum(tf.cast(tf.reshape(pred_masks * gt_masks, [n_pairs, -1]), tf.int32), axis=-1)
    iou = intersects / (area - intersects)
    mask_sm = tf.scatter_nd(mask_ids, iou, sm.shape)

    return mask_sm
