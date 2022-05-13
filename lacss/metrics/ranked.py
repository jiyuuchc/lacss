import tensorflow as tf
import numpy as np
from ..ops import compute_similarity
from ..ops import proposal_locations
from ..ops import IouSimilarity

def np_unique_location_matching(similarity_matrix, threshold):
    ''' Perform matching based on similarity_matrix.
    This is different from the function in ops/matchers.py in that the matching
    will be unique: each column will match at most one row
    Args:
      similarity_matrix: [N, K] tensor, rows are presorted based on scores
      threshold: minimal value to be considered as a match
    Returns:
      matches: [N,] indices[0..K)  of the match for each row
      indicators: [N,] indicator (1/0) value for each match
    '''

    matches = []
    indicators = []
    n, _ = similarity_matrix.shape
    for i_row in range(n):
        row = similarity_matrix[i_row]
        row_max_indice = row.argmax()
        row_max = row[row_max_indice]
        matches.append(row_max_indice)
        if row_max > threshold:
            indicators.append(1)
            similarity_matrix[:, row_max_indice] = -1
        else:
            indicators.append(0)

    return np.array(matches, np.int32), np.array(indicators, np.int32)

def np_compute_ap(similarity_matrix, thresholds):
    # avoid edge cases
    _, k = similarity_matrix.shape
    if k == 0:
        return np.zeros([len(thresholds)], np.float32)

    apmks = []
    for th in thresholds:
        _, indicators = np_unique_location_matching(similarity_matrix, th)
        p_k = np.cumsum(indicators) / (np.arange(len(indicators))+1)
        apmks.append(np.sum(p_k * indicators) / k)
    return np.array(apmks, np.float32)

class BinaryMeanAP(tf.keras.metrics.Metric):
    def __init__(self, thresholds, **kwargs):
        super(BinaryMeanAP, self).__init__(**kwargs)
        self.thresholds = [1.0/(x*x + 1.0e-16) for x in thresholds]

        self.ap_m_k = self.add_weight(name='apmk', dtype=tf.float32, shape=[len(self.thresholds)], initializer='zeros')
        self.total_k = self.add_weight(name='total_k', dtype=tf.int32, initializer='zeros')

    def reset_state(self):
        self.ap_m_k.assign(tf.zeros([len(self.thresholds)], tf.float32))
        self.total_k.assign(tf.constant(0, tf.int32))

    def update_state(self, gt, pred, sample_weight=None):
        sm = compute_similarity(pred, gt)
        apmks = tf.numpy_function(
            lambda s : np_compute_ap(s, self.thresholds),
            [sm],
            tf.float32,
        )

        self.ap_m_k.assign_add(apmks)
        self.total_k.assign_add(1)

    def result(self):
        return tf.reduce_mean(self.ap_m_k / tf.cast(self.total_k, tf.float32))

def similarity_matrix_from_mask_indices(pred_indices, gt_indices):
    gt_indices = tf.cast(gt_indices, tf.int64)
    pred_indices = tf.cast(pred_indices, tf.int64)

    pred_hw = tf.reduce_max(pred_indices.values, axis=0) + 1
    gt_hw = tf.reduce_max(gt_indices.values, axis=0) + 1
    hw = tf.reduce_max(tf.stack([pred_hw, gt_hw]), axis=0)

    mask_shape = (gt_indices.nrows(), hw[0], hw[1])
    idx = tf.concat([gt_indices.value_rowids()[...,None], gt_indices.values], axis=-1)
    all_gt_masks = tf.scatter_nd(idx, tf.ones([tf.shape(idx)[0]], tf.int32), shape=mask_shape)

    def cp_sim_one_row(cur_ind):
        cur_ind = tf.expand_dims(cur_ind, 0)
        cur_ind = tf.tile(cur_ind, [gt_indices.nrows(),1,1])
        gathered = tf.gather_nd(all_gt_masks, cur_ind, batch_dims=1)
        return tf.reduce_sum(gathered, axis=-1)

    intersects = tf.map_fn(
        cp_sim_one_row, pred_indices, fn_output_signature = tf.int32
    )
    intersects = tf.cast(intersects, tf.int64)
    union = gt_indices.row_lengths() + pred_indices.row_lengths()[:,None] - intersects
    iou = intersects / union
    return iou

class BoxMeanAP(tf.keras.metrics.Metric):
    def __init__(self,thresholds=[0.5], **kwargs):
        super(BoxMeanAP, self).__init__(**kwargs)
        self.thresholds = thresholds

        self.ap_m_k = self.add_weight(name='apmk', dtype=tf.float32, shape=[len(self.thresholds)], initializer='zeros')
        self.total_k = self.add_weight(name='total_k', dtype=tf.int32, initializer='zeros')
        self.similarity = IouSimilarity()

    def reset_state(self):
        self.ap_m_k.assign(tf.zeros([len(self.thresholds)], tf.float32))
        self.total_k.assign(tf.constant(0, tf.int32))

    def update_state(self, gt, pred, sample_weight=None):
        sm = self.similarity(pred, gt)
        apmks = tf.numpy_function(
            lambda s : np_compute_ap(s, self.thresholds),
            [sm],
            tf.float32,
        )

        self.ap_m_k.assign_add(apmks)
        self.total_k.assign_add(1)

    def result(self):
        return tf.reduce_mean(self.ap_m_k / tf.cast(self.total_k, tf.float32))

# class MaskMeanAP(tf.keras.metrics.Metric):
#     def __init__(self,thresholds=[0.5], **kwargs):
#         super(MaskMeanAP, self).__init__(**kwargs)
#         self.thresholds = thresholds
#
#         self.ap_m_k = self.add_weight(name='apmk', dtype=tf.float32, shape=[len(self.thresholds)], initializer='zeros')
#         self.total_k = self.add_weight(name='total_k', dtype=tf.int32, initializer='zeros')
#
#     def reset_state(self):
#         self.ap_m_k.assign(tf.zeros([len(self.thresholds)], tf.float32))
#         self.total_k.assign(tf.constant(0, tf.int32))
#
#     def update_state(self, gt, pred, sample_weight=None):
#         sm = similarity_matrix_from_mask_indices(pred, gt)
#         apmks = tf.numpy_function(
#             lambda s : np_compute_ap(s, self.thresholds),
#             [sm],
#             tf.float32,
#         )
#
#         self.ap_m_k.assign_add(apmks)
#         self.total_k.assign_add(1)
#
#     def result(self):
#         return tf.reduce_mean(self.ap_m_k / tf.cast(self.total_k, tf.float32))
#
