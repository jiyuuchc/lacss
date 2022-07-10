import tensorflow as tf
from .unet import UNet
from .resnet import ResNet
from .detector import Detector
from .instance_head import InstanceHead
from .detection_head import DetectionHead
from ..metrics import *
from ..losses import *
from ..ops import *
layers = tf.keras.layers

class LacssModel(tf.keras.Model):
    def __init__(self, backbone, lpn, detector, segmentor,
            train_supervised=False,
            detection_level=3,
            detection_roi_size=1.5,
            loss_weights=(1.0, 1.0, 1.0, 1.0),
            train_batch_size=1,
            ):
        super().__init__()
        self.backbone = backbone
        self.lpn = lpn
        self.detector = detector
        self.segmentor = segmentor
        self.train_supervised = train_supervised
        self.detection_level = detection_level
        self.detection_roi_size = detection_roi_size
        self.loss_weights = loss_weights
        self.train_batch_size = train_batch_size

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
        config_dict = {
            'backbone': self.backbone.__class__.__name__,
            'backbone_config': self.backbone.get_config(),
            'lpn': self.lpn.get_config(),
            'detector': self.detector.get_config(),
            'segmentor': self.segmentor.get_config(),
            'detection_level': self.detection_level,
            'detection_roi_size': self.detection_roi_size,
            'loss_weights': self.loss_weights,
            'train_batch_size': self.train_batch_size,
        }
        return config_dict

    @classmethod
    def from_config(cls, config):
        if 'backbone' in config.keys():
            backbone_name = config.pop('backbone')
            backbone_cfg = config.pop('backbone_config')
        else:
            backbone_name = 'ResNet'
            backbone_cfg = {'model_config': '50'}

        if backbone_name == 'ResNet':
            backbone = ResNet(**backbone_cfg)
        elif backbone_name == 'UNet':
            backbone = UNet(**backbone_cfg)
        else:
            raise ValueError(f'unknown backbone name {backbone_name}')

        if 'lpn' in config.keys():
            lpn = DetectionHead(**config.pop('lpn'))
        else:
            lpn = DetectionHead()

        if 'detector' in config.keys():
            detector = Detector(**config.pop('detector'))
        else:
            detector = Detector()

        if 'segment' in config.keys():
            segmentor = InstanceHead(**config.pop('segmentor'))
        else:
            segmentor = InstanceHead()

        return cls(backbone, lpn, detector, segmentor, **config)

    def build(self, input_shape):
        self._auxnet = [
            layers.Conv2D(24, 3, padding='same', activation='relu', kernel_initializer='he_normal'),
            layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal'),
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


    def get_features(self, imgs, training):
        backbone_out = self.backbone(imgs, training=True)

        lr_features = backbone_out[str(self.detection_level)]

        segment_level = self.segmentor.feature_level
        hr_features = backbone_out[str(segment_level)]

        return lr_features, hr_features

    def call(self, inputs, training=False):
        model_output = {}

        img = inputs['image']
        if len(tf.shape(img)) == 3:
            img = tf.expand_dims(img, 0)

        _, height, width, _ = img.shape
        height = height if height else tf.shape(img)[1]
        width = width if width else tf.shape(img)[2]

        lr_features, hr_features = self.get_features(img, training)

        scores_out, regression_out = self.lpn(lr_features, training=training)
        model_output.update({
            'detection_scores': scores_out,
            'detection_regression': regression_out,
        })

        if training:
            gt_locations = inputs['locations'] / tf.cast([height, width], tf.float32)
        else:
            gt_locations = None
        locations, scores = self.detector((scores_out, regression_out, gt_locations), training=training)

        if not training:
            scaled_locations = locations * tf.cast([height, width], tf.float32)
            model_output.update({
                'proposed_locations': scaled_locations,
                'proposal_location_scores': scores,
            })

        instance_inputs = (hr_features, locations)
        instance_output, instance_coords = self.segmentor(instance_inputs, training=training)
        model_output.update({
            'instance_output': instance_output,
            'instance_coords': instance_coords,
        })

        if training:
            x = img
            for op in self._auxnet:
                x = op(x)
            model_output.update({
                'auxnet_out': x,
            })

        return model_output

    def train_step(self, data):
        height = data['image'].shape[-3]
        width = data['image'].shape[-2]

        with tf.GradientTape() as tape:
            model_output = self(data, training=True)

            score_loss, loc_loss = detection_losses(
                data['locations'] / tf.cast([height, width], tf.float32),
                model_output['detection_scores'],
                model_output['detection_regression'],
                roi_size = self.detection_roi_size,
            )

            instance_loss = 0.
            edge_loss = 0.
            if self.train_supervised:
                for k in range(self.train_batch_size):
                    instance_loss += supervised_segmentation_losses(
                        model_output['instance_output'][k],
                        model_output['instance_coords'][k],
                        data['mask_indices'][k],
                    )
                instance_loss = instance_loss / self.train_batch_size
            else:
                x = model_output['auxnet_out']
                edge_pred = x[:,:,:,0]
                for k in range(self.train_batch_size):
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
                instance_loss = instance_loss / self.train_batch_size
                edge_loss = edge_loss / self.train_batch_size

            weights = self.loss_weights
            loss = score_loss * weights[0] + loc_loss * weights[1] + instance_loss * weights[2] + edge_loss * weights[3]

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients((g, v) for (g, v) in zip(grads, self.trainable_variables) if g is not None)

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
