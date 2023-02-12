from typing import Tuple
from functools import partial
import jax
import treex as tx
jnp = jax.numpy

from .se_net import ChannelAttention
from ..ops import locations_to_labels

class LPN(tx.Module):

    def __init__(
        self,
        feature_levels: Tuple[int] = (4,3,2),
        conv_spec: Tuple[Tuple[int], Tuple[int]] = ((256,256,256,256),()),
        detection_roi: float = 8.0,
    ):
        """
          Args:
            in_channels: number of channels in input feature
            feature_level: input feature level default 3
            conv_layers: conv layer spec
            with_channel_attention: whether include channel attention
        """

        super().__init__()

        self._config_dict = dict(
            feature_levels = feature_levels,
            conv_spec = conv_spec,
            detection_roi = detection_roi,
        )

    @property
    def feature_levels(self):
        return self._config_dict['feature_levels']

    @property
    def detection_roi(self):
        return self._config_dict['detection_roi']

    @tx.compact
    def _process_feature(self, feature: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        conv_spec = self._config_dict['conv_spec']

        x = feature

        for n_ch in conv_spec[0]:
            x = tx.Conv(n_ch, (3,3), use_bias=False)(x)
            # x = tx.BatchNorm()(x, use_running_average=False)
            x = tx.GroupNorm(num_groups=n_ch)(x)
            x = jax.nn.relu(x)

        for n_ch in conv_spec[1]:
            x = tx.Conv(n_ch, (1,1), use_bias=False)(x)
            # x = tx.BatchNorm()(x, use_running_average=False)
            x = tx.GroupNorm(num_groups=n_ch)(x)
            x = jax.nn.relu(x)

        scores_out = tx.Conv(1, (1,1))(x)
        scores_out = jax.nn.sigmoid(scores_out)

        regression_out = tx.Conv(2, (1,1))(x)

        return scores_out, regression_out

    def __call__(self, inputs: dict, scaled_gt_locations: jnp.ndarray = None) -> dict:
        '''
        Args:
            inputs: feature dict: {'lvl': [B, H, W, C]}
            scaled_gt_locations: scaled 0..1 [B, N, 2], only valid in training
        Returns:
            outputs: 
            {
                lpn_scores: dict: {'lvl': [B, H, W, 1]}
                lpn_regressions: dict {'lvl': [B, H, W, 2]}
                gt_lpn_scores: dict {'lvl': [B, H, W, 1]}, only if training
                gt_lpn_regressions: dict {'lvl': [B, H, W, 2]}, only if training
            }
        '''

        all_scores = dict()
        all_regrs = dict()
        all_gt_scores = dict()
        all_gt_regrs = dict()

        for lvl in self.feature_levels:

            feature = inputs[str(lvl)]
            score, regression = self._process_feature(feature)
            all_scores[str(lvl)] = score
            all_regrs[str(lvl)] = regression

            if self.training and scaled_gt_locations is not None:
                op = partial(
                    locations_to_labels, 
                    target_shape = feature.shape[-3:-1],
                    threshold = self.detection_roi / (2 ** lvl),
                )
                score_target, regression_target = jax.vmap(op)(scaled_gt_locations)
                all_gt_scores[str(lvl)] = score_target
                all_gt_regrs[str(lvl)] = regression_target

        outputs = dict(
            lpn_scores = all_scores,
            lpn_regressions = all_regrs,
        )
        if self.training and scaled_gt_locations is not None:
            outputs.update(dict(
                lpn_gt_scores = all_gt_scores,
                lpn_gt_regressions = all_gt_regrs
            ))

        return outputs

    def get_config(self):
        return self._config_dict

    @classmethod
    def from_config(cls, config):
        return(cls(**config))
