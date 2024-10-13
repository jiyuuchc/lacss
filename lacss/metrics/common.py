from __future__ import annotations

import numpy as np

def compute_mask_its(pred_contour, gt_label):
    import cv2
    from skimage.measure import regionprops
    from lacss.ops import box_intersection, box_area

    pred_polygons = pred_contour["pred_contours"]
    pred_bboxes = pred_contour["pred_bboxes"]
    pred_areas = np.array([ cv2.contourArea(ct.astype(int)) for ct in pred_polygons])

    if gt_label.shape[-1] == 1:
        gt_label = gt_label.squeeze(-1)
    dim = gt_label.ndim
    
    rps = regionprops(gt_label)
    gt_bboxes = np.stack([rp.bbox for rp in rps])
    gt_areas = np.stack([rp.area for rp in rps])

    box_its = box_intersection(pred_bboxes, gt_bboxes)

    mask_its = np.zeros_like(box_its, dtype=int)

    h, w = gt_label.shape
    def _get_its(pred_id, gt_id):
        polygon = pred_polygons[pred_id].astype(int)

        if polygon.size > 0:

            img = np.zeros(gt_label.shape, dtype="uint8")
            cv2.fillPoly(img, [polygon], 1)

            gt_coords = rps[gt_id].coords

            return np.count_nonzero(
                img[(gt_coords[:,0], gt_coords[:,1])]
            )

        else:
            return 0        

    ids = np.where(box_its > 0)
    mask_its[ids] = [_get_its(pid, gid) for pid, gid in zip(*ids)]

    return mask_its, pred_areas, gt_areas
