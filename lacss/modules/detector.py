from typing import Optional, Sequence, Union, List, Tuple
import jax
import treex as tx
from ..ops import location_matching, sorted_non_max_suppression
jnp = jax.numpy

class Detector(tx.Module):
    ''' A weightless layer that conver LPN output to a list of locations
    '''

    def __init__(
        self,
        train_nms_threshold: float = 8.0,
        train_pre_nms_topk: int = -1,
        train_max_output: int = 768,
        train_min_score: float = 0.2,
        max_proposal_offset: float = 12,
        test_nms_threshold: float = 8.0,
        test_pre_nms_topk: int = -1,
        test_max_output: int = 768,
        test_min_score:float = 0.25,
    ):
        super().__init__()
        self._config_dict = {
            'train_nms_threshold': train_nms_threshold,
            'train_pre_nms_topk': train_pre_nms_topk,
            'train_max_output': train_max_output,
            'train_min_score': train_min_score,
            'max_proposal_offset': max_proposal_offset,
            'test_nms_threshold': test_nms_threshold,
            'test_pre_nms_topk': test_pre_nms_topk,
            'test_max_output': test_max_output,
            'test_min_score': test_min_score,
        }

    def _proposal_locations(self, lpn_scores, lpn_regression):
        '''
        Produce a list of proposal locations based on predication map, remove redundency with non_max_suppression
        Args:
            lpn_score: dict {'lvl': [height, width, 1]} lpn score prediction
            lpn_regression: dict {'lvl': [height, width, 2]} lpn regression prediction
        Returns:
            scores: [N], sorted scores
            locations: [N, 2], proposed location
        '''
        if self.training:
            distance_threshold = self._config_dict['train_nms_threshold']
            output_size = self._config_dict['train_max_output']
            topk = self._config_dict['train_pre_nms_topk']
            score_threshold = self._config_dict['train_min_score']
        else:
            distance_threshold = self._config_dict['test_nms_threshold']
            output_size = self._config_dict['test_max_output']
            topk = self._config_dict['test_pre_nms_topk']
            score_threshold = self._config_dict['test_min_score']

        # preprocess
        scores = []
        locations = []
        for k in lpn_scores:
            s = lpn_scores[k].squeeze(-1)
            r = lpn_regression[k]
            height, width = s.shape
            loc = jnp.mgrid[:height, :width] + .5
            loc = loc.transpose(1,2,0) + r

            # set invalid location score to 0
            is_valid = (loc > 0.0).all(axis=-1) & (loc[:,:,0] < height) & (loc[:,:,1] < width)
            s = jnp.where(is_valid, s, 0)

            loc *= 2**int(k) # scale to origianl image size

            locations.append(loc.reshape(-1, 2))
            scores.append(s.reshape(-1))

        scores = jnp.concatenate(scores, axis=0)
        locations = jnp.concatenate(locations, axis=0)

        # select topk
        if topk <= 0 or topk > scores.size:
            topk = scores.size
        scores, selections = jax.lax.top_k(scores, topk)
        locations = locations[selections]

        # refine
        if distance_threshold > 0: # nms
            threshold = 1 / distance_threshold / distance_threshold
            scores, locations = sorted_non_max_suppression(
                scores, locations, output_size, threshold, score_threshold,
            )
        else:
            is_valid = scores >= score_threshold
            scores = jnp.where(is_valid, scores, -1.)
            locations = jnp.where(is_valid[:, None], locations, -1.)

        return scores, locations

    def _gen_train_locations(self, gt_locations, pred_locations):
        ''' replacing gt_locations with pred_locations if the close enough
            gt_locations: [N, 2] tensor
            pred_locations: [M, 2] tensor, sorted with scores

            1. Each pred_location is matched to the closest gt_location
            2. For each gt_location, pick the matched pred_location with highest score
            3. if the picked pred_location is within threshold distance, replace the gt_location with the pred_location
        '''

        threshold = self._config_dict['max_proposal_offset']

        n_gt_locs = gt_locations.shape[0]
        n_pred_locs = pred_locations.shape[0]

        matched_id, indicators = location_matching(pred_locations, gt_locations, threshold)
        matched_id = jnp.where(indicators, matched_id, -1)
        
        matching_matrix = matched_id[None, :] == jnp.arange(n_gt_locs)[:, None]  # true at matched gt(row)/pred(col), at most one true per col
        last_col = jnp.ones([n_gt_locs, 1], dtype=bool)
        matching_matrix = jnp.concatenate([matching_matrix, last_col], axis=-1) # last col is true

        # first true of every row 
        matched_loc_ids = jnp.argmax(matching_matrix, axis=-1)

        training_locations = jnp.where(
            matched_loc_ids[:, None] == n_pred_locs,  # true: failed match
            gt_locations,
            pred_locations[matched_loc_ids, :] # out-of-bound error silently dropped in jax
        )

        return training_locations

    def __call__(
        self, 
        scores: jnp.ndarray,
        regressions: jnp.ndarray, 
        gt_locations: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Args:
            score: {'lvl', [B, H, W, 1]) output from LPN 
            regression: ('lvl': [B, H, W, 2]) output from LPN
            gt_locations: [B, N, 2] during training or None during inference. Maybe padded with -1
        Outputs: 
            if training:
            dict {
                training_locations: [B, N, 2] 
            }
            if interfernce:
            dict {
                pred_locations: [B, M, 2] sorted based on scores; M == test_max_output; padded with -1
                pred_scores: [B, M] sorted, padded with -1
            }
        '''

        scores = jax.lax.stop_gradient(scores)
        regressions = jax.lax.stop_gradient(regressions)

        if self.training and self._config_dict['max_proposal_offset'] <= 0:
            return dict(
                training_locations = gt_locations,
            )

        proposed_scores, proposed_locations = jax.vmap(self._proposal_locations)(
            scores, 
            regressions,
        )
        outputs = dict(
                pred_locations = proposed_locations,
                pred_scores = proposed_scores,
            )
        if self.training:
            outputs.update(dict(
                training_locations = jax.vmap(self._gen_train_locations)(
                    gt_locations,
                    proposed_locations,
                )
            ))
        return outputs

    def get_config(self):
        return self._config_dict

    @classmethod
    def from_config(cls, config):
        return(cls(**config))
