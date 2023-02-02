import tensorflow as tf
import numpy as np
from ..ops import *

class AJI:
  def __init__(self, th=.5):
      self.c = 0
      self.u = 0
      self.th = th

  def update(self, gt_mi, gt_box, pred_patches, pred_patch_co):
      gt_areas = gt_mi.row_lengths().numpy()
      pred_areas = tf.math.count_nonzero(pred_patches > self.th, axis=(1,2,3)).numpy()
      sum_areas = gt_areas[:, None] + pred_areas

      intersects = mask_intersects(gt_mi, gt_box, pred_patches, pred_patch_co, self.th).numpy()
      ious = intersects / (sum_areas - intersects + 1e-8)
      best_matches = ious.argmax(axis=1)
      best_intersects = np.take_along_axis(intersects, best_matches[:, None], axis=1)
      all_intersects = best_intersects.sum()

      self.u += all_intersects

      self.c += gt_areas.sum()
      self.c += pred_areas[best_matches].sum()
      self.c -= all_intersects
      pred_areas[best_matches] = 0
      self.c += pred_areas.sum()

  def result(self):
      return self.u / self.c
