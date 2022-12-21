import tensorflow as tf
from .iou_similarity import *
from ..ops import *

def has_box_intersection(gt_box, pred_box):
    gt_box = tf.cast(gt_box, tf.float32)
    pred_box = tf.cast(pred_box, tf.float32)

    y_min1, x_min1, y_max1, x_max1 = tf.split(gt_box, 4, -1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(pred_box, 4, -1)

    y_min_max = tf.minimum(y_max1, tf.transpose(y_max2))
    y_max_min = tf.maximum(y_min1, tf.transpose(y_min2))
    x_min_max = tf.minimum(x_max1, tf.transpose(x_max2))
    x_max_min = tf.maximum(x_min1, tf.transpose(x_min2))

    intersect_heights = y_min_max - y_max_min
    intersect_widths = x_min_max - x_max_min

    return (intersect_heights > 0) & (intersect_widths > 0)

def mask_intersects(gt_mi, gt_box, pred_patches, pred_patch_co, th=0.5):
    pred_box = bboxes_of_patches(pred_patches, pred_patch_co, th)
    ids = tf.where(has_box_intersection(gt_box, pred_box))
    gt_ids = ids[:, 0]
    pred_ids = ids[:, 1]

    pred_patches = tf.gather(pred_patches>th, pred_ids)
    pred_patches = tf.squeeze(pred_patches, -1)
    y0x0 = tf.gather(pred_patch_co[:, 0, 0, :], pred_ids)
    gt_mi_x = tf.gather(gt_mi, gt_ids)
    y0x0 = tf.repeat(y0x0, gt_mi_x.row_lengths(), axis=0)
    fns = gt_mi_x.value_rowids()[..., None]
    values = gt_mi_x.values - tf.cast(y0x0, gt_mi_x.dtype)
    gt_mi_flat = tf.concat([fns, values], axis=-1)
    gt_patches = tf.scatter_nd(gt_mi_flat, tf.ones(values.shape[0], dtype=tf.float32), pred_patches.shape)
    gt_patches = tf.cast(gt_patches, tf.bool)

    its = tf.math.count_nonzero(gt_patches & pred_patches, axis=(1,2))
    intersects = tf.scatter_nd(ids, its, shape=[gt_box.shape[0], pred_box.shape[0]])
    return intersects


def mask_ious(gt_mi, gt_box, pred_patches, pred_patch_co, th=0.5):
    gt_areas = gt_mi.row_lengths()
    pred_areas = tf.count_nonzero(pred_patches > th, axis=(1,2,3))
    sum_areas = gt_areas[:, None] + pred_areas
    intersects = mask_intersects(gt_mi, gt_box, pred_patches, pred_patch_co, th)
    ious = tf.cast(intersects, tf.float32) / (tf.cast(sum_areas - intersects, tf.float32) + 1e-8)
    return ious

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
