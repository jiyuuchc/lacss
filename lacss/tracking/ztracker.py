from typing import Sequence

from dataclasses import dataclass
import logging

import numpy as np
import scipy

from .kalman import KalmanFilter, KTracker
from ..ops import iou_loss, box_iou_similarity, generalized_iou_loss

def _from_obs(yxhw):
    return np.r_[yxhw[:2] - yxhw[2:]/2 , yxhw[:2] + yxhw[2:]/2]

def _to_obs(yxyx):
    return np.r_[(yxyx[:2] + yxyx[2:])/2 , (yxyx[2:] - yxyx[:2])]

def _nms_primary(boxes, scores, score_threshold, iou_threshold):
    ious = box_iou_similarity(boxes, boxes)
    selected = scores > score_threshold
    for irow in range(scores.shape[0]):
        if selected[irow]:
            supress = ious[irow] > iou_threshold
            supress[:(irow + 1)] = False
            selected[supress] = False
    return selected

def _nms_secondary(selected_boxes, boxes, iou_threshold):
    ious = box_iou_similarity(selected_boxes, boxes)
    supress = (ious > iou_threshold).any(axis=0)
    selected = ~ supress
    ious = box_iou_similarity(boxes, boxes)
    for irow in range(ious.shape[0]):
        if selected[irow]:
            supress = ious[irow] > iou_threshold
            supress[:(irow + 1)] = False
            selected[supress] = False
    return selected


@dataclass
class ZStackTracker:
    """ A special tracker for building 3d segmentation from a z-stack of 2d segmentations. It uses a kalman
    tracker of 2d boxes that assumes a fixed center location and tracking only the size changes.

    Attributes:
        score_thresh: score threshold for high-confidence detections
        nms_iou: filter the 2D results with non-max-supression
        cost_thresh: cost threshold for link assignment
        cost_thresh_secondary:  cost threshold for link assignment of low confidence detections
        max_init_area: maximum segmentation area to be considered as the first slice of a cell 
        max_time_lost: max consecutive missing slices  
        std_weight_position: kalman filter parameter. relative error for position 
        std_weight_velocity: kalman filter parameter. relative error for velocity
        min_z_slices: minimal z slices needed to be considered a cell
        use_generalized_iou_loss: whether to use generalized iou loss instead of simple iou_loss for linking
            cost
    """
    score_thresh: float = 0.5 # threshold for high-confidence detections
    nms_iou: float = 0.3
    cost_thresh: float = 0.65 # cost threshold for link assignment
    cost_thresh_secondary: float = 0.5 # cost threshold for link assignment of low confidence detections
    max_init_area: float = 1000
    max_time_lost: int = 1
    std_weight_position: float = 1/20
    std_weight_velocity: float = 1/160
    min_z_slices: int = 3
    use_generalized_iou_loss: bool = False

    # tracks y, x, h, w, vh, vw
    class ZStack_KF(KalmanFilter):
        def __init__(self, std_weight_position, std_weight_velocity):
            self._motion_mat = np.eye(6, 6)
            self._motion_mat[2, 4] = 1
            self._motion_mat[3, 5] = 1
            self._update_mat = np.eye(4, 6)
            self._std_weight_position = std_weight_position
            self._std_weight_velocity = std_weight_velocity

        def initiate(self, measurement):
            mean_pos = measurement
            mean_vel = np.zeros([2])
            mean = np.r_[mean_pos, mean_vel]
            std = [
                2 * self._std_weight_position * measurement[2],
                2 * self._std_weight_position * measurement[3],
                2 * self._std_weight_position * measurement[2],
                2 * self._std_weight_position * measurement[3],
                10 * self._std_weight_velocity * measurement[2],
                10 * self._std_weight_velocity * measurement[3],
            ]
            covariance = np.diag(np.square(std))

            return mean, covariance

        def predict(self, mean, covariance):
            mean = np.dot(mean, self._motion_mat.T)

            std = [
                0, # self._std_weight_position * mean[2],
                0, # self._std_weight_position * mean[3],
                self._std_weight_position * mean[2],
                self._std_weight_position * mean[3],
                self._std_weight_velocity * mean[2],
                self._std_weight_velocity * mean[3],
            ]
            covariance = np.linalg.multi_dot((
                self._motion_mat, covariance, self._motion_mat.T)
            ) + np.diag(np.square(std))

            return mean, covariance

        def project(self, mean, covariance):
            std = [
                0, #self._std_weight_position * mean[2],
                0, #self._std_weight_position * mean[3],
                self._std_weight_position * mean[2],
                self._std_weight_position * mean[3],
            ]
            innovation_cov = np.diag(np.square(std))

            mean = np.dot(self._update_mat, mean)
            covariance = np.linalg.multi_dot((
                self._update_mat, covariance, self._update_mat.T
            ))
            return mean, covariance + innovation_cov

        def update(self, mean, covariance, measurement):
            projected_mean, projected_cov = self.project(mean, covariance)

            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
                check_finite=False).T
            innovation = measurement - projected_mean

            new_mean = mean + np.dot(innovation, kalman_gain.T)
            new_covariance = covariance - np.linalg.multi_dot((
                kalman_gain, projected_cov, kalman_gain.T))

            new_mean[:2] = (mean[:2] + new_mean[:2]) / 2
            new_covariance[:2, :2] = (new_covariance[:2, :2] + projected_cov[:2, :2])/4

            return new_mean, new_covariance

    def __post_init__(self):
        self._kf = self.ZStack_KF(std_weight_position=self.std_weight_position, std_weight_velocity=self.std_weight_velocity)


    def _assign(self, tracks, dets, threshold):
        box_a = np.asarray([_from_obs(t.mean[:4]) for t in tracks]).reshape(-1, 4)
        box_b = np.asarray([_from_obs(t.obs) for t in dets]).reshape(-1, 4)

        if self.use_generalized_iou_loss:
            cost_matrix = generalized_iou_loss(box_a, box_b)
        else:   
            cost_matrix = iou_loss(box_a, box_b)

        return KTracker.assign(tracks, dets, cost_matrix, threshold)

    def update(self, tracks:Sequence[KTracker], dets:dict, frame_id: int) -> tuple[list[KTracker], list[KTracker]]:
        """ Update the tracker with one frame

        Args:
            tracks: a list of KTacker representing currently tracks cells
            dets: predictor output in bbox format
            frame_id: current frame number

        Returns:
            tracked_tracks: list of tracks that are active
            removed_tracks: list of tracks that are inactive

        """
        scores = np.asarray(dets["pred_scores"]).reshape(-1)
        bboxes = np.asarray(dets["pred_bboxes"]).reshape(-1, 4)
        assert len(scores) == bboxes.shape[0]
        assert bboxes.shape[1] == 4
        flatten_dets = np.asarray([dict(zip(dets.keys(), x)) for x in zip(*dets.values())], dtype=object)
        detections = np.asarray([ 
            KTracker(
                _to_obs(box),
                frame_id,
                self._kf,
                data = data,
            ) for box, data in zip(bboxes, flatten_dets)
        ], dtype=object)

        logging.info(f"processing frame {frame_id}: {len(tracks)} input tracks and {len(flatten_dets)} new detections.")

        inds_primary = _nms_primary(bboxes, scores, self.score_thresh, self.nms_iou)
        inds_secondary = ~ inds_primary

        # find current pool of tracks
        removed_tracks, track_pool = [], []
        for track in tracks:
            if track.state == "tracked" or track.state == "lost":
                track.predict()
                track_pool.append(track)
            else:
                raise ValueError(f"Invalid track state {track.state}.")

        ''' First association, with high score detection boxes'''
        tracked_tracks, track_pool, detections_primary = self._assign(
            track_pool, 
            detections[inds_primary],
            threshold=self.cost_thresh,
        )

        ''' Second association, with low score detection boxes'''
        lost_tracks, r_track_pool = [], []
        for track in track_pool:
            if track.state == "tracked":
                r_track_pool.append(track)
            else:
                lost_tracks.append(track)

        cur_boxes = bboxes[inds_primary]
        inds_secondary = _nms_secondary(cur_boxes, bboxes, self.nms_iou)

        matched, unmatched, detections_secondary = self._assign(
            r_track_pool, 
            detections[inds_secondary], 
            threshold=self.cost_thresh_secondary,
        )
        tracked_tracks += matched
        lost_tracks += unmatched

        n_tracks = len(tracked_tracks)
        logging.info(f"processing frame {frame_id}: found {n_tracks} connections")

        """ remove tracks that have been lost for too long """
        for track in lost_tracks:
            if frame_id - track.frame_id > self.max_time_lost:
                lost_tracks.remove(track)
                removed_tracks.append(track)

        logging.info(f"processing frame {frame_id}: remove {len(removed_tracks)} tracks that has lost for too long")

        """ Init new stracks"""
        cur_boxes = np.asarray([t.history[-1]["pred_bboxes"] for t in tracked_tracks]).reshape(-1, 4)
        inds_remaining = _nms_secondary(cur_boxes, bboxes, self.nms_iou)
        for track in detections[inds_remaining]:
            if track._data["pred_scores"] >= 0.5:
                if track.obs[2:].prod() < self.max_init_area:
                    track.initialize()
                    tracked_tracks.append(track)

        logging.info(f"processing frame {frame_id}: initialize {len(tracked_tracks) - n_tracks} new detections")

        # sanity check
        cur_boxes = np.asarray([t.history[-1]["pred_bboxes"] for t in tracked_tracks]).reshape(-1, 4)
        ious = box_iou_similarity(cur_boxes, cur_boxes)
        ious[(range(len(cur_boxes)), range(len(cur_boxes)))] = 0
        assert not ( ious > self.nms_iou ).any()

        """ Update state"""
        for t in lost_tracks:
            m = t.mean
            m[-2:] = 0 # clear velocity for lost tracks
            t.mark_lost(update_state=m)

        return tracked_tracks + lost_tracks, removed_tracks

    def finalize(self, tracks: list[KTracker], *, fill_in_missing:bool=True) -> list[KTracker]:
        """ Finalize step which compute scores and bboxes, and optionally fill in the missing segmentations.
        The aggregated score is the mean of the top-k 2d scores, where k is the min_z_slices. 

        Args:
            tracks: list of tracks, each represent a cell in 3d
        
        Keyword Args:
            fill_in_missing: whether try to generate segmentations for missing slices

        Returns:
            list of tracks will "score", "bbox" fields.        
        """
        tracks = [track for track in tracks if track.frame_id - track.start_frame + 1 >= self.min_z_slices]

        if fill_in_missing:
            for track in tracks:
                data0 = track.history[0]
                f0 = data0["frame_id"] - 1
                for idx, data in enumerate(track.history):
                    if data["frame_id"] != f0 + 1: # missing frame
                        new_data = data.copy()
                        f0 = f0 + 1
                        new_data.update(dict(
                            frame_id = f0,
                            pred_scores = -1,
                            pred_bboxes = (data0["pred_bboxes"] + data["pred_bboxes"])/2,
                            pred_masks = (data["pred_masks"] + data0["pred_masks"]) / 2
                        ))
                        track.history.insert(idx, new_data)
                    else:
                        f0 = data["frame_id"]
                        data0 = data
        
        for cell_data in tracks:
            z0 = cell_data.start_frame
            z1 = cell_data.frame_id + 1
            scores, bboxes = [], []
            for slice_data in cell_data.history:
                scores.append(max(slice_data["pred_scores"], 0.))
                bboxes.append(slice_data["pred_bboxes"])
            bboxes = np.stack(bboxes)
            scores = np.stack(scores)
            top_score_idx = np.argpartition(scores, -self.min_z_slices)[-self.min_z_slices:]
            cell_data.score = scores[top_score_idx].mean()
            cell_data.bbox = np.r_[z0, bboxes[:, :2].min(axis=0), z1, bboxes[:, 2:].max(axis=0)]            

        return tracks

    @staticmethod
    def render_label(tracks:list[KTracker], img3d_shape:Sequence[int])->np.ndarray:
        """ create 3d label from tracking results

        Args:
            tracks: list of tracks
            img3d_shape: tuple of (D, H, W)

        Returns:
            label: array of shape (D, H, W) 
        """
        from skimage.transform import resize
        label = np.zeros(img3d_shape, dtype="uint8")
        _, h, w = img3d_shape

        score_idx = np.argsort([t.score for t in tracks])

        for c, track_id in enumerate(score_idx):
            track = tracks[track_id]
            for data in track.history:
                box = np.round(data["pred_bboxes"]).astype(int)
                box = np.minimum([h,w,h,w], box)
                box = np.maximum(0, box)
                mask = resize(data["pred_masks"], box[2:] - box[:2])
                mask = (mask >= 0.5) * (c+1)
                f = data["frame_id"]
                label[f, box[0]:box[2], box[1]:box[3]] = np.maximum(mask, label[f, box[0]:box[2], box[1]:box[3]])
        
        return label
