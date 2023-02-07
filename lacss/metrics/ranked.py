import numpy as np
import jax
import treex as tx
# import jax.experimental.host_callback as hcb
jnp = jax.numpy

'''
metrics classes for computing coco-style AP metrics.

BE CAREFUL making changes here. Very easy to make a mistake and resulting mismatch with
coco algorithm. Be sure to validate against coco-evaluation before commit changes.
'''

def _unique_location_matching(similarity_matrix, threshold):
    ''' Perform matching based on similarity_matrix.
    This is different from the matchre functions in ops/*.py in that the matching
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
        _, indicators = _unique_location_matching(similarity_matrix, th)
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

        self._result = np.array([-1.0] * len(thresholds))

    def update_state(self, sm, scores):
        # sm = self.similarity(pred, gt)
        self.cell_counts += sm.shape[1]
        self.scores.append(scores)
        for th, indicators in zip(self.thresholds, self.indicator_list):
            _, ind = _unique_location_matching(sm, th)
            indicators.append(ind)

        self._result = None

    def result(self):
        if self._result is not None:
            return self._result

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

        self._result = np.array(aps, dtype=float)

        return self._result

# class LoiAP(tx.Metric):
#     def __init__(self, thresholds, **kwargs):
#         super().__init__(**kwargs)
#         self.ap = MeanAP(thresholds)

#     def update(self, preds, gt_locations, **kwargs):

#         def _update_cb(args, transform):
#             pred, gt_locs = args
#             scores = np.asarray(pred['pred_scores'])
#             mask = scores > 0
#             gt_locs = np.asarray(gt_locs)
#             pred_locs = np.asarray(pred['pred_locations'])[mask]
#             dist2 = ((pred_locs[:,None,:] - gt_locs) ** 2).sum(axis=-1)
#             sm = 1.0 / np.sqrt(dist2)
#             self.ap.update_state(sm, scores[mask])

#         pred = jax.tree_map(lambda v: v[0], preds)
#         hcb.id_tap(_update_cb, (pred, gt_locations))
    
#     def compute_cb(self):
#         return self.ap.result()

#     def compute(self):
#         hcb.barrier_wait()
#         def _compute_cb(arg):
#             return self.ap.result()
        
#         return hcb.call(
#             _compute_cb,
#             None,
#             result_shape=
#         )


