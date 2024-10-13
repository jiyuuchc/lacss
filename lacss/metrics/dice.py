from __future__ import annotations

import numpy as np

class Dice:
    """Compute instance Dice values"""

    def __init__(self):
        self.pred_areas = []
        self.gt_areas = []
        self.pred_scores = []
        self.gt_scores = []

    def update(self, pred_its, pred_areas, gt_areas):
        self.pred_areas.append(pred_areas)
        self.gt_areas.append(gt_areas)

        pred_best = pred_its.max(axis=1)
        pred_best_matches = pred_its.argmax(axis=1)
        pred_dice = pred_best * 2 / (pred_areas + gt_areas[pred_best_matches])
        self.pred_scores.append(pred_dice)

        gt_best = pred_its.max(axis=0)
        gt_best_matches = pred_its.argmax(axis=0)
        gt_dice = gt_best * 2 / (gt_areas + pred_areas[gt_best_matches])
        self.gt_scores.append(gt_dice)

    def compute(self):
        pred_areas = np.concatenate(self.pred_areas)
        gt_areas = np.concatenate(self.gt_areas)
        pred_scores = np.concatenate(self.pred_scores)
        gt_scores = np.concatenate(self.gt_scores)

        pred_dice = (pred_areas / pred_areas.sum() * pred_scores).sum()
        gt_dice = (gt_areas / gt_areas.sum() * gt_scores).sum()

        dice = (pred_dice + gt_dice) / 2

        return dice

