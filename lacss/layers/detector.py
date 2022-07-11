import tensorflow as tf
from ..ops import *

layers = tf.keras.layers

class Detector(layers.Layer):
    ''' A weightless layer that conver LPN output to a list of locations
    '''

    def __init__(self,
            # detection_level=3,
            # detection_roi_size=1.5,
            detection_nms_threshold=1.0,
            train_pre_nms_topk=2000,
            train_max_output=500,
            train_min_score=0.2,
            max_proposal_offset=0.03,
            test_pre_nms_topk=2000,
            test_max_output=500,
            test_min_score=0.5,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._config_dict = {
            # 'detection_level': detection_level,
            # 'detection_roi_size': detection_roi_size,
            'detection_nms_threshold': detection_nms_threshold,
            'train_pre_nms_topk': train_pre_nms_topk,
            'train_max_output': train_max_output,
            'train_min_score': train_min_score,
            'max_proposal_offset': max_proposal_offset,
            'test_pre_nms_topk': test_pre_nms_topk,
            'test_max_output': test_max_output,
            'test_min_score': test_min_score,
        }
        self._config_dict.update(kwargs)

    def get_config(self):
        return self._config_dict

    def _gen_train_locations(self, batched_gt_locations, batched_pred_locations):
        threshold = self._config_dict['max_proposal_offset']

        def _training_locations_for_one_image(gt_locations, pred_locations):
            ''' replacing gt_locations with pred_locations if the close enough
              gt_locations: [N, 2] tensor
              pred_locations: [M, 2] tensor, sorted with scores

              1. Each pred_location is matched to the closest gt_location
              2. For each gt_location, pick the matched pred_location with highest score
              3. if the picked pred_location is within threshold distance, replace the gt_location with the pred_location
            '''
            n_gt_locs = tf.shape(gt_locations)[0]
            n_pred_locs = tf.shape(pred_locations)[0]
            matched_id, indicators = location_matching_unpadded(pred_locations, gt_locations, [threshold], [0,1])
            matched_id = tf.where(tf.cast(indicators, tf.bool), matched_id, -1)
            matching_matrix = matched_id == tf.range(n_gt_locs)[:,None]
            matching_matrix = tf.concat([matching_matrix, tf.ones([n_gt_locs,1], tf.bool)], axis=-1)
            all_locs = tf.where(matching_matrix)
            matched_loc_ids = tf.math.segment_min(all_locs[:,1], all_locs[:,0])
            matched_loc_ids = tf.cast(matched_loc_ids, n_pred_locs.dtype)

            matched_locs = tf.gather(pred_locations, matched_loc_ids)
            # matched_locs = tf.where(matched_loc_ids[:,None]==n_pred_locs, gt_locations, matched_locs)

            max_allowed_pred = n_pred_locs
            # max_allowed_pred = n_gt_locs + 10

            training_locations = tf.where(
                matched_loc_ids[:,None]>=max_allowed_pred,
                gt_locations,
                matched_locs,
                )

            return training_locations

        return tf.map_fn(
            lambda x: _training_locations_for_one_image(*x),
            (batched_gt_locations, batched_pred_locations),
            fn_output_signature=tf.RaggedTensorSpec([None,2], batched_gt_locations.dtype, 0),
        )

    def call(self, inputs, training=None):
        scores_out, regression_out, gt_locations = inputs

        if training:
            max_output = self._config_dict['train_max_output']
            topk = self._config_dict['train_pre_nms_topk']
            min_score = self._config_dict['train_min_score']
        else:
            max_output = self._config_dict['test_max_output']
            topk = self._config_dict['test_pre_nms_topk']
            min_score = self._config_dict['test_min_score']

        proposed_scores, proposed_locations = proposal_locations(
                scores_out, regression_out,
                max_output_size=max_output,
                distance_threshold=self._config_dict['detection_nms_threshold'],
                topk=topk,
                score_threshold=min_score,
                )

        if training:
            proposed_locations = self._gen_train_locations(
                gt_locations,
                proposed_locations,
                )

        return proposed_locations, proposed_scores
