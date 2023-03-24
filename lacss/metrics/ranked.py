import jax
import jax.numpy as jnp
import numpy as np

from ..ops import *
from ..train.metric import Metric
from ..utils import _get_name

"""
metrics classes for computing coco-style AP metrics.

BE CAREFUL making changes here. Very easy to make a mistake and resulting mismatch with
coco algorithm. Be sure to validate against coco-evaluation before commit changes.
"""


def _unique_location_matching(similarity_matrix, threshold):
    """Perform matching based on similarity_matrix.
    This is different from the matchre functions in ops/*.py in that the matching
    will be unique: each column will match at most one row
    Args:
      similarity_matrix: [N, K] tensor, rows are presorted based on scores
      threshold: minimal value to be considered as a match
    Returns:
      matches: [N,] indices[0..K)  of the match for each row
      indicators: [N,] indicator (1/0) value for each match
    """

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
        p_k = np.cumsum(indicators) / (np.arange(len(indicators)) + 1)
        apmks.append(np.sum(p_k * indicators) / k)

    return np.array(apmks, np.float32)


class AP(Metric):
    """compute AP based on similarity_matrix and score.
    These are numpy functions
    Usage:
      m = MeanAP([threshold_1, threshold_2,...])
      m.update_states(similarity_matrix, scores)
      ...
      m.result()
    """

    def __init__(self, thresholds=[0.5], coco_style=False, name=None):
        self.thresholds = thresholds
        self.coco_style = coco_style
        if name is None:
            self.name = _get_name(self)
        else:
            self.name = name
        self.reset()

    def update(self, sm, scores):
        self.cell_counts += sm.shape[1]
        self.scores.append(scores)
        for th, indicators in zip(self.thresholds, self.indicator_list):
            _, ind = _unique_location_matching(sm, th)
            indicators.append(ind)

        self._result = None

    def compute(self):
        if self._result is not None:
            return self._result

        scores = np.concatenate(self.scores)
        indices = np.argsort(scores)
        aps = []
        for indicators in self.indicator_list:
            indicators = np.concatenate(indicators)
            indicators = indicators[indices[::-1]]
            p_k = np.cumsum(indicators) / (np.arange(len(indicators)) + 1)
            if self.coco_style:
                p_k = np.maximum.accumulate(p_k[::-1])[::-1]
            aps.append(np.sum(p_k[indicators == 1]) / self.cell_counts)

        self._result = np.array(aps, dtype=float)

        return self._result

    def reset(self):
        self.scores = []
        self.cell_counts = 0
        self.indicator_list = [[] for _ in range(len(self.thresholds))]
        self._result = np.array([-1.0] * len(self.thresholds))


class LoiAP(AP):
    def __init__(self, thresholds=[0.5], **kwargs):
        super().__init__([th * th for th in thresholds], **kwargs)

    def update(self, preds, gt_locations, **kwargs):
        scores = preds["pred_scores"]
        pred_locations = preds["pred_locations"]
        n_batch = gt_locations.shape[0]
        for k in range(n_batch):
            score = scores[k]
            pred = pred_locations[k]
            gt = gt_locations[k]
            row_mask = score > 0
            col_mask = (gt >= 0).all(axis=-1)
            dist2 = ((pred[:, None, :] - gt[:, :]) ** 2).sum(axis=-1)

            dist2 = np.asarray(1.0 / dist2)
            dist2 = dist2[row_mask][:, col_mask]
            score = np.asarray(score)[row_mask]

            super().update(dist2, score)


class BoxAP(AP):
    def update(self, preds, gt_boxes, **kwargs):
        pred_boxes = jax.vmap(bboxes_of_patches)(preds)
        box_ious = np.asarray(box_iou_similarity(pred_boxes, gt_boxes))
        scores = np.asarray(preds["pred_scores"])
        valid_gt_boxes = np.asarray((gt_boxes >= 0).all(axis=-1))

        for score, iou, gt_is_valid in zip(scores, box_ious, valid_gt_boxes):
            valid_preds = score >= 0
            iou = iou[valid_preds][:, gt_is_valid]
            score = score[valid_preds]
            super().update(iou, score)


class MaskAP(AP):
    def _update(self, pred, gt_label):
        scores = np.asarray(pred["pred_scores"])
        ious = np.asarray(iou_patches_and_labels(pred, gt_label))

        valid = scores >= 0
        valid_scores = scores[valid]
        valid_ious = ious[valid]
        super().update(valid_ious, valid_scores)

    def update(self, preds, gt_labels, **kwargs):

        for k in range(len(gt_labels)):
            self._update(
                *jax.tree_util.tree_map(
                    lambda v: v[k],
                    (preds, gt_labels),
                )
            )
