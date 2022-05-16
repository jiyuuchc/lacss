import tensorflow as tf
import tensorflow.keras.layers as layers
from .detection_head import DetectionHead
from .unet import UNetDecoder, UNetEncoder
from .instance_head import InstanceHead
from ..metrics import *
from ..losses import *
from ..ops import *
from .unet import *
from .resnet import *

class LacssModel(tf.keras.Model):
    def __init__(self,
            backbone = 'unet_s',
            detection_head_conv_filers=(1024,),
            detection_head_fc_filers=(1024,),
            detection_roi_size=1.5,
            detection_nms_threshold=1.0,
            train_pre_nms_topk=2000,
            train_max_output=500,
            train_min_score=0,
            test_pre_nms_topk=0,
            test_max_output=500,
            test_min_score=0,
            # instance_jitter=0.,
            max_proposal_offset=16,
            instance_crop_size=96,
            instance_n_convs=1,
            instance_conv_channels=64,
            train_supervied=False,
            ):
        super().__init__()
        self._config_dict = {
            'backbone': backbone,
            'detection_head_conv_filers':detection_head_conv_filers,
            'detection_head_fc_filers':detection_head_fc_filers,
            'detection_roi_size': detection_roi_size,
            'detection_nms_threshold': detection_nms_threshold,
            'train_pre_nms_topk': train_pre_nms_topk,
            'train_max_output': train_max_output,
            'train_min_score': train_min_score,
            'test_pre_nms_topk': test_pre_nms_topk,
            'test_max_output': test_max_output,
            'test_min_score': test_min_score,
            # 'instance_jitter':instance_jitter,
            'max_proposal_offset': max_proposal_offset,
            'instance_crop_size': instance_crop_size,
            'instance_n_convs': instance_n_convs,
            'instance_conv_channels': instance_conv_channels,
            'train_supervied': train_supervied,
        }

        self._metrics = [
            tf.keras.metrics.Mean('loss', dtype=tf.float32),
            tf.keras.metrics.Mean('score_loss', dtype=tf.float32),
            tf.keras.metrics.Mean('localization_loss', dtype=tf.float32),
            tf.keras.metrics.Mean('instance_loss', dtype=tf.float32),
            BinaryMeanAP([10.0], name='mAP'),
            BoxMeanAP(name='box_mAP'),
        ]

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        backbone = self._config_dict['backbone']
        if backbone == 'unet_s':
            self._backbone = build_unet_s_backbone()
        elif backbone == 'unet':
            self._backbone = build_unet_backbone()
        elif backbone == 'resnet':
            self._backbone = build_resnet_backbone()
        else:
            raise ValueError(f'unknown backbone type: {backbone}')

        self._detection_head = DetectionHead(
            conv_filters=self._config_dict['detection_head_conv_filers'],
            fc_filters=self._config_dict['detection_head_fc_filers'],
        )
        self._instance_head = InstanceHead(
              n_patch_conv_layers=self._config_dict['instance_n_convs'],
              n_conv_channels=self._config_dict['instance_conv_channels'],
              instance_crop_size=self._config_dict['instance_crop_size']//2,
              )

    def _update_metrics(self, new_metrics):
        logs={}
        for m in self._metrics:
            if m.name in new_metrics:
                new_data = new_metrics[m.name]
                if type(new_data) is tuple:
                    m.update_state(*new_data)
                else:
                    m.update_state(new_data)
                logs.update({m.name: m.result()})
        return logs

    @property
    def metrics(self):
        return self._metrics

    def _gen_train_locations(self, gt_locations, pred_locations):
        threshold = self._config_dict['max_proposal_offset']
        # instance_jitter = self._config_dict['instance_jitter']
        # if  instance_jitter > 0:
        #     jitter = tf.random.uniform(tf.shape(inputs['locations']), maxval=instance_jitter) - instance_jitter/2
        # else:
        #     jitter = 0

        n_gt_locs = tf.shape(gt_locations)[0]
        n_pred_locs = tf.shape(pred_locations)[0]
        matched_id, indicators = location_matching_unpadded(pred_locations, gt_locations, [threshold], [0,1])
        matched_id = tf.where(tf.cast(indicators, tf.bool), matched_id, -1)
        matching_matrix = matched_id == tf.range(n_gt_locs)[:,None]
        matching_matrix = tf.concat([matching_matrix, tf.ones([n_gt_locs,1], tf.bool)], axis=-1)
        all_locs = tf.where(matching_matrix)
        matched_loc_ids = tf.math.segment_min(all_locs[:,1], all_locs[:,0])
        matched_loc_ids=tf.cast(matched_loc_ids, n_pred_locs.dtype)

        matched_locs = tf.gather(pred_locations, matched_loc_ids)
        matched_locs = tf.where(matched_loc_ids[:,None]==n_pred_locs, gt_locations, matched_locs)

        training_locations = tf.where(
            matched_loc_ids[:,None]==n_pred_locs,
            # gt_locations + jitter,
            gt_locations,
            matched_locs,
            )

        return training_locations

    def call(self, inputs, training=False):
        model_output = {}

        img = inputs['image']
        height, width, _ = img.shape
        height = height if height else tf.shape(img)[0]
        width = width if width else tf.shape(img)[1]

        y = tf.expand_dims(img, 0)
        segmentation_features, detection_features = self._backbone(y, training=training)
        detection_features = tf.squeeze(detection_features, 0)
        segmentation_features = tf.squeeze(segmentation_features, 0)

        scores_out, regression_out = self._detection_head(detection_features, training=training)

        if training:
            scaled_gt_locations = inputs['locations'] / [height, width]
            model_output.update({
                'detection_scores': scores_out,
                'detection_regression': regression_out,
                'scaled_gt_locations': scaled_gt_locations,
            })
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
        decoded_locations = proposed_locations * [height, width]
        model_output.update({
            'pred_location_scores': proposed_scores,
            'pred_locations': decoded_locations,
            })

        if training:
            training_locations = self._gen_train_locations(
                inputs['locations'],
                decoded_locations,
                )
            instance_inputs = (segmentation_features, training_locations / [height, width])
        else:
            instance_inputs = (segmentation_features, proposed_locations)

        instance_output, instance_coords = self._instance_head(instance_inputs, training=training)

        model_output.update({
            'instance_output': instance_output,
            'instance_coords': instance_coords,
        })

        return model_output

    def train_step(self, data):
        with tf.GradientTape() as tape:
            model_output = self(data, training=True)

            score_loss, loc_loss = detection_losses(
                model_output['scaled_gt_locations'][None, ...],
                model_output['detection_scores'][None, ...],
                model_output['detection_regression'][None, ...],
                roi_size = self._config_dict['detection_roi_size'],
            )

            if self._config_dict['train_supervied']:
                instance_loss = supervised_segmentation_losses(
                    model_output['instance_output'],
                    model_output['instance_coords'],
                    data['mask_indices'],
                )
                instance_loss /= 500.0
            else:
                instance_loss = self_supervised_segmentation_losses(
                    model_output['instance_output'],
                    model_output['instance_coords'],
                    data['binary_mask'],
                    )

            loss = score_loss + loc_loss + instance_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients((g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None)

        gt_locations = data['locations']
        pred_locations = model_output['pred_locations']
        logs = self._update_metrics({
            'loss': loss,
            'score_loss': score_loss,
            'localization_loss': loc_loss,
            'instance_loss': instance_loss,
            'mAP': (gt_locations, pred_locations),
        })

        return logs

    def test_step(self, data):
        model_output = self(data, training=False)

        gt_bboxes = data['bboxes']
        pred_bboxes = bboxes_of_patches(
            model_output['instance_output'],
            model_output['instance_coords'],
            )
        # gt_locations = tf.reduce_mean(tf.reshape(gt_bboxes, [-1,2,2]), axis=-2)
        # pred_locations = model_output['pred_locations']
        logs = self._update_metrics({
            # 'mAP': (gt_locations, pred_locations),
            'box_mAP': (gt_bboxes, pred_bboxes),
        })

        return logs
