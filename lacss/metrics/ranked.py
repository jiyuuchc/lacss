import tensorflow as tf
import numpy as np
from ..ops import compute_similarity
from ..ops import proposal_locations

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
    def __init__(self, thresholds, image_format=False, **kwargs):
        super(BinaryMeanAP, self).__init__(**kwargs)
        self.thresholds = [1.0/(x*x + 1.0e-16) for x in thresholds]
        self.image_format = image_format

        self.ap_m_k = self.add_weight(name='apmk', dtype=tf.float32, shape=[len(self.thresholds)], initializer='zeros')
        self.total_k = self.add_weight(name='total_k', dtype=tf.int32, initializer='zeros')

    def reset_state(self):
        self.ap_m_k.assign(tf.zeros([len(self.thresholds)], tf.float32))
        self.total_k.assign(tf.constant(0, tf.int32))

    def update_state(self, gt_locations, pred_locations, sample_weight=None):
        if self.image_format:
            pred_scores, pred_locations = proposal_locations(pred_locations)

        for k in range(pred_locations.nrows()):
            pred = pred_locations[k]
            gt = gt_locations[k]
            if self.image_format:
                if len(gt.shape) == 3:
                    gt = tf.squeeze(gt, -1)
                gt = tf.cast(tf.where(gt > 0), tf.int32)
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
