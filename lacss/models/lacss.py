import tensorflow as tf
from .instance_head import InstanceHead
from .detection_head import DetectionHead
from ..metrics import *
from ..losses import *
from ..ops import *
from .unet import *
from .resnet import *
layers = tf.keras.layers

class LacssModel(tf.keras.Model):
    def __init__(self,
            backbone = 'unet_s',
            detection_head_conv_filers=(1024,),
            detection_head_fc_filers=(1024,),
            detection_level=3,
            detection_roi_size=1.5,
            detection_nms_threshold=1.0,
            train_pre_nms_topk=2000,
            train_max_output=500,
            train_min_score=0,
            test_pre_nms_topk=0,
            test_max_output=500,
            test_min_score=0,
            max_proposal_offset=16,
            instance_crop_size=96,
            instance_n_convs=1,
            instance_conv_channels=64,
            train_supervised=False,
            loss_weights=(1.0, 1.0, 1.0, 1.0),
            train_batch_size=1,
            ):
        super().__init__()
        self._config_dict = {
            'backbone': backbone,
            'detection_head_conv_filers':detection_head_conv_filers,
            'detection_head_fc_filers':detection_head_fc_filers,
            'detection_level': detection_level,
            'detection_roi_size': detection_roi_size,
            'detection_nms_threshold': detection_nms_threshold,
            'train_pre_nms_topk': train_pre_nms_topk,
            'train_max_output': train_max_output,
            'train_min_score': train_min_score,
            'test_pre_nms_topk': test_pre_nms_topk,
            'test_max_output': test_max_output,
            'test_min_score': test_min_score,
            'max_proposal_offset': max_proposal_offset,
            'instance_crop_size': instance_crop_size,
            'instance_n_convs': instance_n_convs,
            'instance_conv_channels': instance_conv_channels,
            'train_supervised': train_supervised,
            'loss_weights': loss_weights,
            'train_batch_size': train_batch_size,
        }

        self._metrics = [
            tf.keras.metrics.Mean('loss', dtype=tf.float32),
            tf.keras.metrics.Mean('score_loss', dtype=tf.float32),
            tf.keras.metrics.Mean('localization_loss', dtype=tf.float32),
            tf.keras.metrics.Mean('instance_loss', dtype=tf.float32),
            tf.keras.metrics.Mean('edge_loss', dtype=tf.float32),
            LOIMeanAP([10.0], name='loi_mAP'),
            BoxMeanAP(name='box_mAP'),
        ]

    def get_config(self):
        return self._config_dict

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        backbone = self._config_dict['backbone']
        img_shape = (None, None, input_shape['image'][-1])
        if backbone == 'resnet2':
            self._backbone = build_resnet_backbone(input_shape=img_shape, is_v2=True, with_attention=False)
        elif backbone == 'resnet2_att':
            self._backbone = build_resnet_backbone(input_shape=img_shape, is_v2=True, with_attention=True)
        elif backbone == 'resnet':
            self._backbone = build_resnet_backbone(input_shape=img_shape, with_attention=False)
        elif backbone == 'resnet_att':
            self._backbone = build_resnet_backbone(input_shape=img_shape, with_attention=True)
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

        if not self._config_dict['train_supervised']:
            if backbone == 'resnet2' or backbone =='resnet2_att':
                self._edge_predictor = [
                    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal'),
                    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal'),
                    layers.Conv2D(1, 3, padding='same', activation='sigmoid', kernel_initializer='he_normal'),
                ]
            else:
                self._edge_predictor = [
                    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal'),
                    layers.Conv2D(1, 3, padding='same', activation='sigmoid', kernel_initializer='he_normal'),
                ]

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

    def call(self, inputs, training=False):
        model_output = {}

        img = inputs['image']
        if len(tf.shape(img)) == 3:
            img = tf.expand_dims(img, 0)
        _, height, width, _ = img.shape
        height = height if height else tf.shape(img)[1]
        width = width if width else tf.shape(img)[2]

        y = img
        encoder_out, decoder_out = self._backbone(y, training=True)
        # detection_features = tf.squeeze(decoder_out[-2], 0)
        # segmentation_features = tf.squeeze(decoder_out[-4], 0)
        detection_level = self._config_dict['detection_level']
        detection_features = decoder_out[detection_level-1]
        segmentation_features = decoder_out[0]
        stem_features = encoder_out[0]

        scores_out, regression_out = self._detection_head(detection_features, training=training)

        model_output.update({
            'detection_scores': scores_out,
            'detection_regression': regression_out,
            'stem_features': stem_features,
        })

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
            training_locations = (training_locations/[height, width]).to_tensor(-1.0)
        else:
            training_locations = proposed_locations.to_tensor(-1.0)
        instance_inputs = (segmentation_features, training_locations)

        instance_output, instance_coords = self._instance_head(instance_inputs, training=training)

        model_output.update({
            'instance_output': instance_output,
            'instance_coords': instance_coords,
        })

        return model_output

    def train_step(self, data):
        height = data['image'].shape[-3]
        width = data['image'].shape[-2]

        with tf.GradientTape() as tape:
            model_output = self(data, training=True)

            score_loss, loc_loss = detection_losses(
                data['locations'] / [height, width],
                # model_output['scaled_gt_locations'][None, ...],
                model_output['detection_scores'],
                model_output['detection_regression'],
                roi_size = self._config_dict['detection_roi_size'],
            )

            instance_loss = 0.
            edge_loss = 0.
            if self._config_dict['train_supervised']:
                for k in range(self._config_dict['train_batch_size']):
                    instance_loss += supervised_segmentation_losses(
                        model_output['instance_output'][k],
                        model_output['instance_coords'][k],
                        data['mask_indices'][k],
                    )
                instance_loss = instance_loss / self._config_dict['train_batch_size']
            else:
                x = model_output['stem_features']
                for layer in self._edge_predictor:
                    x = layer(x)
                edge_pred = x[:,:,:,0]
                for k in range(self._config_dict['train_batch_size']):
                    instance_loss += self_supervised_segmentation_losses(
                        model_output['instance_output'][k],
                        model_output['instance_coords'][k],
                        data['binary_mask'][k],
                    )
                    edge_loss += self_supervised_edge_losses(
                        model_output['instance_output'][k],
                        model_output['instance_coords'][k],
                        edge_pred[k],
                    )
                instance_loss = instance_loss / self._config_dict['train_batch_size']
                edge_loss = edge_loss / self._config_dict['train_batch_size']

            weights = self._config_dict['loss_weights']
            loss = score_loss * weights[0] + loc_loss * weights[1] + instance_loss * weights[2] + edge_loss * weights[3]

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients((g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None)

        # gt_locations = data['locations']
        # pred_locations = model_output['pred_locations']
        # scores = model_output['pred_location_scores']
        logs = self._update_metrics({
            'loss': loss,
            'score_loss': score_loss,
            'localization_loss': loc_loss,
            'instance_loss': instance_loss,
            'edge_loss': edge_loss,
        })

        return logs

    # def test_step(self, data):
    #     model_output = self(data, training=False)
    #
    #     gt_bboxes = data['bboxes']
    #     pred_bboxes = bboxes_of_patches(
    #         model_output['instance_output'],
    #         model_output['instance_coords'],
    #         )
    #     scores = model_output['pred_location_scores']
    #     gt_locations = data['locations']
    #     pred_locations = model_output['pred_locations']
    #     logs = self._update_metrics({
    #         'loi_mAP': (gt_locations, pred_locations, scores),
    #         'box_mAP': (gt_bboxes, pred_bboxes, scores),
    #     })
    #
    #     return logs
