import tensorflow as tf
import tensorflow.keras.layers as layers
from .detection_head import DetectionHead
from ..metrics import *
from ..losses import *
from ..ops import *

class LacssModel(tf.keras.Model):
    def __init__(self,
            backbone, detection_head, instance_head,
            detection_roi_size=1.5,
            detection_pre_nms_topk=2000,
            detection_nms_threshold=1.1,
            detection_max_output=500,
            max_proposal_offset=16,):
        super().__init__()
        self._config_dict = {
            'backbone': backbone,
            'detection_head': detection_head,
            'instance_head': instance_head,
            'detection_roi_size': detection_roi_size,
            'detection_pre_nms_topk': detection_pre_nms_topk,
            'detection_nms_threshold': detection_nms_threshold,
            'detection_max_output': detection_max_output,
            'max_proposal_offset': max_proposal_offset,
        }

        self._metrics = [
            tf.keras.metrics.Mean('loss', dtype=tf.float32),
            tf.keras.metrics.Mean('score_loss', dtype=tf.float32),
            tf.keras.metrics.Mean('localization_loss', dtype=tf.float32),
            tf.keras.metrics.Mean('binary_mask_loss', dtype=tf.float32),
            tf.keras.metrics.Mean('instance_loss', dtype=tf.float32),
            BinaryMeanAP([12.0], False, name='mAP'),
            tf.keras.metrics.BinaryAccuracy(name='mask_acc')
        ]

        self._mask_layer = layers.Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal', name='mask')

    def get_config(self):
        return self._config_dict

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
        matched_id, indicators = location_matching(gt_locations, pred_locations, [threshold,], [0, 1])

        matched_locations = tf.gather_nd(pred_locations, matched_id[...,None], batch_dims=1)

        training_locations = tf.where(indicators[...,None] > 0, matched_locations, gt_locations)

        return training_locations

    def call(self, inputs, training=False):
        model_output = {}

        backbone = self._config_dict['backbone']
        detection_head = self._config_dict['detection_head']
        instance_head = self._config_dict['instance_head']

        backbone_out = backbone(inputs['image'], training=training)
        detection_features = backbone_out['detection_features']
        segmentation_features = backbone_out['segmentation_features']

        scores_out, regression_out = detection_head(detection_features, training=training)
        if training:
            model_output.update({
                'detection_scores': scores_out,
                'detection_regression': regression_out,
            })

        pred_scores, pred_locations = proposal_locations(
                scores_out, regression_out,
                max_output_size=self._config_dict['detection_max_output'],
                distance_threshold=self._config_dict['detection_nms_threshold'],
                topk=self._config_dict['detection_pre_nms_topk'],
                )
        scale = tf.cast(tf.shape(inputs['image'])[1:3], pred_locations.dtype)
        pred_locations = pred_locations * scale
        model_output.update({
            'location_scores': pred_scores,
            'locations': pred_locations,
        })

        if training:
            mask = self._mask_layer(segmentation_features, training=training)
            model_output.update({
                'mask': mask,
            })

        if training:
            training_locations = self._gen_train_locations(inputs['locations'], pred_locations.to_tensor(-1))
        else:
            training_locations = pred_locations.to_tensor(-1)

        training_locations = training_locations / tf.cast(tf.shape(inputs['image'])[1:3], training_locations.dtype)
        instance_inputs = (segmentation_features, detection_features, training_locations)
        instance_output, instance_coords = instance_head(instance_inputs, training=training)

        if training:
            model_output.update({
                'instance_output': instance_output,
                'instance_coords': instance_coords,
            })

        return model_output

    def train_step(self, data):
        detection_roi_size = self._config_dict['detection_roi_size']

        gt_locations = data['locations']
        gt_mask = data['binary_label']

        with tf.GradientTape() as tape:
            model_output = self(data, training=True)
            scaled_gt_locations = gt_locations / tf.cast(tf.shape(data['image'])[1:3], gt_locations.dtype)
            score_loss, loc_loss = detection_losses(
                scaled_gt_locations,
                model_output['detection_scores'],
                model_output['detection_regression'],
                threshold = detection_roi_size,
            )

            # pred_mask = model_output['mask']
            # mask_loss = tf.keras.losses.binary_crossentropy(gt_mask, pred_mask)

            instance_output = model_output['instance_output']
            instance_coords = model_output['instance_coords']
            # instance_loss = tf.map_fn(
            #     lambda x: self_supervised_segmentation_losses(x[0], x[1], x[2]),
            #     (instance_output, gt_mask, instance_coords),
            #     fn_output_signature=tf.float32,
            # )
            # instance_loss = tf.reduce_sum(instance_loss)
            instance_loss = self_supervised_segmentation_losses(instance_output[0], gt_mask[0], instance_coords[0])
            #loss = score_loss + loc_loss + mask_loss + instance_loss
            loss = score_loss + loc_loss + instance_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients((g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None)

        pred_locations = model_output['locations']
        logs = self._update_metrics({
            'loss': loss,
            'score_loss': score_loss,
            'localization_loss': loc_loss,
            # 'binary_mask_loss': mask_loss,
            'instance_loss': instance_loss,
            'mAP': (gt_locations, pred_locations),
            # 'mask_acc': (gt_mask, pred_mask),
        })

        return logs

    def test_step(self, data):
        gt_locations = data['locations']
        #gt_mask = data['mask_label']
        #gt_mask = tf.cast(gt_mask > 0, tf.int32)
        #_, mask_height, mask_width =  gt_mask.shape.as_list()
        # gt_mask = tf.expand_dims(gt_mask, -1)

        model_output = self(data, training=True)
        pred_locations = model_output['locations']
        #pred_mask = model_output['mask'] #scaled

        #target_size = tf.cast(tf.cast(tf.shape(pred_mask)[1:3], tf.float32) / data['scaling_factor'], tf.int32)
        #pred_mask = tf.image.resize(pred_mask, target_size)
        #pred_mask = tf.image.crop_to_bounding_box(pred_mask, 0, 0, mask_height, mask_width)

        logs = self._update_metrics({
            'mAP': (gt_locations, pred_locations),
            # 'mask_acc': (gt_mask, pred_mask),
        })

        return logs
