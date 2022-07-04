import tensorflow as tf
import numpy as np
from ..ops import *

'''
metrics classes for computing coco-style AP metrics.

BE CAREFUL making changes here. Very easy to make a mistake and resulting mismatch with
coco algorithm. Be sure to validate against coco-evaluation before commit changes.
'''

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
    similarity_matrix = np.copy(similarity_matrix)
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

class MeanAP():
    ''' compute mAP based on similarity_matrix and score.
      These are numpy functions
      Usage:
        m = MeanAP([threshold_1, threshold_2,...])
        m.update_state(similarity_matrix, scores)
        ...
        m.result()
    '''
    def __init__(self,thresholds=[0.5], coco_style=False):
        self.thresholds = thresholds
        self.cell_counts = 0
        self.scores = []
        self.coco_style=coco_style
        self.indicator_list = [[] for _ in range(len(thresholds))]

    def update_state(self, sm, scores):
        # sm = self.similarity(pred, gt)
        self.cell_counts += sm.shape[1]
        self.scores.append(scores)
        for th, indicators in zip(self.thresholds, self.indicator_list):
            _, ind = np_unique_location_matching(sm, th)
            indicators.append(ind)
        return self.cell_counts

    def result(self):
        scores = np.concatenate(self.scores)
        indices = np.argsort(scores)
        aps = []
        for indicators in self.indicator_list:
            indicators = np.concatenate(indicators)
            indicators = indicators[indices[::-1]]
            p_k = np.cumsum(indicators) / (np.arange(len(indicators))+1)
            if self.coco_style:
                p_k = np.maximum.accumulate(p_k[::-1])[::-1]
            aps.append(np.sum(p_k[indicators==1]) / self.cell_counts)

        return np.array(aps, dtype=np.float32)

class BoxMeanAP(tf.keras.metrics.Metric):
    '''Compute coco-style mean AP based on bbox iou.
      Usage:
        m = BoxMeanAP([threshold_1, threshold_2,...])
        m.update_state(gt_bboxes, prediction_bboxes, prediction_scores)
        ...
        m.result()
    '''
    def __init__(self,thresholds=[0.5], **kwargs):
        super(BoxMeanAP, self).__init__(**kwargs)
        self.thresholds = thresholds
        self.similarity = IouSimilarity()

        self.reset_state()

    def reset_state(self):
        self._np_mean_aps = MeanAP(self.thresholds)

    def update_state(self, gt, pred, scores):
        sm = self.similarity(pred, gt)
        tf.numpy_function(
            self._np_mean_aps.update_state,
            [sm, scores],
            tf.int64,
        )

    def result(self):
        aps = tf.numpy_function(
            self._np_mean_aps.result,
            [],
            tf.float32,
        )
        return aps

class LOIMeanAP(BoxMeanAP):
    def __init__(self, thresholds, **kwargs):
        super(LOIMeanAP, self).__init__(**kwargs)
        self.thresholds = [1.0/(x*x + 1.0e-16) for x in thresholds]
        self.similarity = compute_similarity
        self.reset_state()

class MaskMeanAP(BoxMeanAP):
    '''Compute coco-style mean AP based on segmentation iou.
      Usage:
        m = MaskMeanAP([threshold_1, threshold_2,...])
        # gt_mask_indices: RaggedTensor of [n_instances, None, 2]
        # gt_bboxes: Tensor of [n_instances, 4]
        gt = (gt_mask_indices, gt_bboxes)
        # instances: Tensor of [n_predictions, patch_size, patch_size, 1]
        # coordinates: Tensor of [n_predictions, patch_size, patch_size, 2]
        # bboxes: Tensor of [n_predictions, 4] or None
        prediction = (instances, coordinates, bboxes)
        m.update_state(gt, prediction, prediction_scores)
        ...
        m.result()
    '''
    def __init__(self, thresholds, **kwargs):
        super(MaskMeanAP, self).__init__(thresholds, **kwargs)

    def _get_bboxes(self, mask_indices):
        c0 = tf.reduce_min(mask_indices, axis=1)
        c1 = tf.reduce_max(mask_indices, axis=1) + 1
        bboxes = tf.concat([c0, c1], axis=-1)
        return bboxes

    def update_state(self, gt, pred, scores):
        pred_instances, pred_coords, pred_bboxes = pred
        patch_size = pred_instances.shape[1]
        if pred_bboxes is None:
            pred_bboxes = bboxes_of_patches(pred_instances, pred_coords)
        pred_mask_indices = indices_of_patches(pred_instances, pred_coords)

        gt_mask_indices, gt_bboxes = gt
        if gt_bboxes is None:
          gt_bboxes = self._get_bboxes(gt_mask_indices)

        sm = mask_iou_similarity(
            (gt_mask_indices, gt_bboxes),
            (pred_mask_indices, pred_bboxes),
            patch_size=patch_size,
            )

        tf.numpy_function(
            self._np_mean_aps.update_state,
            [sm, scores],
            tf.int64,
        )

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
