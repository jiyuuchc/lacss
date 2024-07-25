from __future__ import annotations

from typing import Protocol, Any, Sequence
import math

import numpy as np
import scipy

from .utils import linear_assignment
from ..ops import iou_loss
from ..typing import ArrayLike


class KalmanFilter(Protocol):
    """KalmanFilter Protocol definition"""

    def initialize(self, measurement: Any)->tuple[np.ndarray, np.ndarray]:
        """Create track from unassociated measurement.

        Args:
            measurement : data in measurement space

        Returns:
            mean: Unobserved velocities are initialized to 0 mean.
            cov: covariance matrix (8x8 dimensional) of the new track. 

        """
        ...
    
    def predict(self, mean: Any, covariance: Any)->tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step.

        Args:
            mean : mean vector of the object state at the previous time step.
            covariance : The covariance matrix of the object state at the
                previous time step.

        Returns:
            mean: the mean vector of the predicted state
            cov: the covariance matrix of the predicted
        """
        ...

    def update(self, mean: Any, covariance: Any, measurement: Any)->tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter correction step.

        Args:
            mean :  The predicted state's mean vector.
            covariance : The state's covariance matrix.
            measurement : The measurement vector

        Returns:
            mean: mean vector of the measurement-corrected state distribution.
            cov: covariance matrix of the measurement-corrected state distribution.
        """
        ...


class ConstantVelocityKalmanFilter(KalmanFilter):
    """
    A simple Kalman filter for tracking bounding boxes in image space.  The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self, *, std_weight_position:float = 1/20, std_weight_velocity:float =1/160):
        """
        Keyword Args:
            std_weight_position: Relative observation uncertainty.
            std_weight_velocity: Relative velocity uncertainty.
        """
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

    def initiate(self, measurement:ArrayLike)->tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            2 * 1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * 1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean:ArrayLike, covariance:ArrayLike)->tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean:ArrayLike, covariance:ArrayLike) ->tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean:ArrayLike, covariance:ArrayLike)->tuple[np.ndarray, np.ndarray]:
        """vectorized version of the prediction step (Vectorized version).

        Args:
            mean : Nx8 dimensional mean matrix of the object states at the previous
                time step.
            covariance : Nx8x8 dimensional covariance matrics of the object states at the
                previous time step.

        Returns:
            mean: the mean vector of the predicted state
            cov:  the covariance matrix of the predicted state
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean:ArrayLike, covariance:ArrayLike, measurement:ArrayLike)->tuple[np.ndarray, np.ndarray]:
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

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position:bool|None=False, metric:str='maha'
        ) -> np.ndarray:
        """ Compute gating distance between state distribution and measurements.
            A suitable distance threshold can be obtained from `chi2inv95`. If
            `only_position` is False, the chi-square distribution has 4 degrees of
            freedom, otherwise 2.
        
        Args:
            mean: Mean vector over the state distribution (8 dimensional).
            covariance : Covariance of the state distribution (8x8 dimensional).
            measurements : An Nx4 dimensional matrix of N measurements, each in
                format (x, y, a, h) where (x, y) is the bounding box center
                position, a the aspect ratio, and h the height.
            only_position : If True, distance computation is done with respect to the bounding
                box center position only.

        Returns: 
            gating_distance: an array of length N, where the i-th element contains the
                squared Mahalanobis distance between (mean, covariance) and `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')


class ConstantVelocityKalmanFilter3D(KalmanFilter):
    """A 3D Kalman filter for tracking 3D bounding boxes follows a constant velocity model. 
        The 8-dimensional state space are:

            z, y, x, d, h, w, vz, vy, vx, vd, vh, vw
    """
    def __init__(self, *, std_weight_position:float=1/20, std_weight_velocity:float=1/160, z_scaling:float=1):
        """
        Keyword Args:
            std_weight_position: Relative observation uncertainty.
            std_weight_velocity: Relative velocity uncertainty.
            z_scaling: relative pixel size along z-axis vs x-y-axis
        """
        ndim, dt = 6, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = np.array([
            std_weight_position / z_scaling,
            std_weight_position,
            std_weight_position,
            std_weight_position / z_scaling,
            std_weight_position,
            std_weight_position,
        ])            

        self._std_weight_velocity = np.array([
            std_weight_velocity / z_scaling,
            std_weight_velocity,
            std_weight_velocity,
            std_weight_velocity / z_scaling,
            std_weight_velocity,
            std_weight_velocity
        ]) 

    def initiate(self, measurement:ArrayLike)->tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        mean_size = (math.prod(mean[4:6])) ** (1/2)
        std = np.r_[
            2 * self._std_weight_position * mean_size,
            10 * self._std_weight_velocity * mean_size,
        ]
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean:ArrayLike, covariance:ArrayLike)->tuple[np.ndarray, np.ndarray]:
        mean_size = (math.prod(mean[4:6])) ** (1/2)

        std_pos = self._std_weight_position * mean_size
        std_vel = self._std_weight_velocity * mean_size

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean:ArrayLike, covariance:ArrayLike)->tuple[np.ndarray, np.ndarray]:
        mean_size = (math.prod(mean[4:6])) ** (1/2)

        std = self._std_weight_position * mean_size

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean:ArrayLike, covariance:ArrayLike)->tuple[np.ndarray, np.ndarray]:
        """Vectorized Kalman filter prediction step.
        """
        mean_size = np.sqrt(mean[:, -1] * mean[:, -2])

        std_pos = self._std_weight_position[:, None] * mean_size
        std_vel = self._std_weight_velocity[:, None] * mean_size

        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean:ArrayLike, covariance:ArrayLike, measurement:ArrayLike)->tuple[np.ndarray, np.ndarray]:
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

        return new_mean, new_covariance

class KTracker:
    """ A stateful KalmanFilter tracker
    """
    _count = -1
 
    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

    @property
    def mean(self):
        try:
            return self._mean.copy()
        except AttributeError:
            return None

    @property
    def cov(self):
        try:
            return self._cov.copy()
        except AttributeError:
            return None
    

    def __repr__(self):
        if self.state == "uninitialized":
            return f"KT_U_{self.frame_id}"
        else:
            return f'KT_{self.track_id}_({self.start_frame}-{self.frame_id})'


    def __init__(self, init_obs:ArrayLike, frame_id:int, kf: KalmanFilter, *, data:dict={}):
        """ constructor

        Args:
            init_obs: initial measurement space data
            frame_id: the frame number
            kf: a Kalman Filter following the KalmanFilter protocol
        
        Keyword Args:
            data: additional data of the observation that will be store
        """
        self.obs = np.asarray(init_obs, dtype=float)
        self._data = data
        self.tracklet_len = 0
        self.kf = kf
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.state = "uninitialized"


    def predict(self):
        """ Kalman filter prediction
        """
        self._mean, self._cov = self.kf.predict(
            self.mean, self.cov,
        )


    def initialize(self):
        """ Initialze the track, so that it has a track id and a history object
        """
        self._mean, self._cov = self.kf.initiate(
            self.obs,
        )            
        self.track_id = self.next_id()
        self._data.update(dict(
            frame_id = self.frame_id,
            predicted_obs = self.mean,
        ))
        self.history = [self._data]
        self.state = "tracked"


    def update(self, new_track: KTracker):
        """ Kalman filter update step

        new_track: the new observation
        """
        self._mean, self._covariance = self.kf.update(
            self.mean, self.cov, new_track.obs
        )

        self.frame_id = new_track.frame_id
        self.obs = new_track.obs

        if self.state == "tracked":
            self.tracklet_len += 1
        else:
            self.tracklet_len = 0
        
        self.state = "tracked"
        data = new_track._data
        data.update(dict(
            frame_id = new_track.frame_id,
            preicted_obs = self.mean,
        )) 
        self.history.append(new_track._data)


    def new_id(self):
        self.track_id = self.next_id() 


    def mark_lost(self, *, update_state:ArrayLike=None):
        """ Mark the track as lost

        Keyword Args:
            update_state: if not None, change the mean vector to this one
        """
        self.state = "lost"
        if update_state is not None:
            self._mean = np.array(update_state)


    def mark_removed(self):
        """ Mark the track as removed
        """
        self.state = "removed"

    @staticmethod
    def assign(
        tracks: Sequence[KTracker], 
        dets: Sequence[KTracker], 
        cost_matrix: ArrayLike,
        threshold:float,
    ) -> tuple[Sequence[KTracker], Sequence[KTracker], Sequence[KTracker]]:
        """ perform MOT assigment

        Args:
            tracks: current list of tracks of length M
            dets: list of new observations of length N
            cost_matrix: [M x N] cost matrix
            threshold: max cost for linking
        
        Returns:
            tracked: list of tracks that are updated with new observations
            remaining_tracks: list of tracks that did not find a suitable link
            remaining_dets: the remaining detections that didn't get linked in.            
        """
        matches, unmatched_track_ids, unmatched_det_ids = linear_assignment(cost_matrix, thresh=threshold)

        tracked = []
        for itracked, idet in matches:
            track = tracks[itracked]
            track.update(dets[idet])
            tracked.append(track)            

        remaining_tracks = [tracks[it] for it in unmatched_track_ids]
        remaining_dets = [dets[it] for it in unmatched_det_ids]

        return tracked, remaining_tracks, remaining_dets
