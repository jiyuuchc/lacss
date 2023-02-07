import jax
import treex as tx
from . import ResNet, ConvNeXt, XCiT, Segmentor, LPN, AuxNet, LPN, UNet, Detector
from ..ops import *
jnp = jax.numpy

class Lacss(tx.Module):
    backbone: tx.Module
    lpn: tx.Module
    detector: tx.Module
    segmentor: tx.Module
    auxnet: tx.Module

    def __init__(self, 
            backbone, lpn, detector, segmentor, auxnet, **kwargs,
            ):
        super().__init__()
        self.backbone = backbone
        self.lpn = lpn
        self.detector = detector
        self.segmentor = segmentor
        self.auxnet = auxnet

    def get_config(self):
        config_dict = {
            'backbone': self.backbone.__class__.__name__,
            'backbone_config': self.backbone.get_config(),
            'lpn': self.lpn.get_config(),
            'detector': self.detector.get_config(),
            'segmentor': self.segmentor.get_config(),
            'auxnet': self.auxnet.get_config() ,
        }

        return config_dict

    @classmethod
    def from_config(cls, config):
        if 'backbone' in config.keys():
            backbone_name = config.pop('backbone')
            backbone_cfg = config.pop('backbone_config')
        else:
            backbone_name = 'ResNet'
            backbone_cfg = {'model_spec': '50'}

        if backbone_name == 'ResNet':
            backbone = ResNet(**backbone_cfg)
        elif backbone_name == 'UNet':
            backbone = UNet(**backbone_cfg)
        elif backbone_name == 'XCiT':
            backbone = XCiT(**backbone_cfg)
        elif backbone_name == 'convnext':
            backbone = ConvNeXt(**backbone_cfg)
        else:
            raise ValueError(f'unknown backbone name {backbone_name}')

        if 'lpn' in config:
            lpn = LPN(**config.pop('lpn'))
        else:
            lpn = LPN()

        if 'detector' in config:
            detector = Detector(**config.pop('detector'))
        else:
            detector = Detector()

        if 'segmentor' in config:
            segmentor = Segmentor(**config.pop('segmentor'))
        else:
            segmentor = Segmentor()

        if 'auxnet' in config:
            auxnet = AuxNet(**config.pop('auxnet'))
        else:
            auxnet = AuxNet()

        return cls(backbone, lpn, detector, segmentor, auxnet, **config)

    def _compute_edge(self, instance_output, instance_yc, instance_xc, height, width):
        ps = self.segmentor._config_dict['instance_crop_size']
        padding = ps // 2 + 1

        patch_edges = jnp.square(sorbel_edges(instance_output.squeeze(-1)))
        patch_edges = (patch_edges[0] + patch_edges[1]) / 8.0
        patch_edges = jnp.sqrt(jnp.clip(patch_edges, 1e-8, 1.0)) # avoid GPU error
        # patch_edges = jnp.where(patch_edges > 0, jnp.sqrt(patch_edges), 0)
        combined_edges = jnp.zeros([height + padding*2, width + padding*2])
        combined_edges = combined_edges.at[instance_yc + padding, instance_xc + padding].add(patch_edges)
        combined_edges = combined_edges[padding:padding+height, padding:padding+width]
        combined_edges = jnp.tanh(combined_edges)
        return combined_edges

    @tx.compact
    def __call__(
        self, 
        image: jnp.ndarray, 
        gt_locations: jnp.ndarray = None,
    ) -> dict:
        '''
        Args:
            image: [H, W, C]
            gt_locations: [N, 2] if training, otherwise None
        Returns:
            a dict of model outputs
            if training:
                'detection_scores': LPN output
                'detection_regression': LPN output
                'training_locations': training locations
                'instance_output': instance segmentations
                'instance_y_coords: 
                'instance_x_coords: 
                'auxnet_out':
            else:
                'pred_locations': predicted cell locations,
                'pred_location_scores': scores,
                'instance_output': instance segmentations
                'instance_y_coords: 
                'instance_x_coords: 
        '''
        _, height, width, _ = image.shape
        if gt_locations is not None:
            scaled_gt_locations = gt_locations / jnp.array([height, width])
        else:
            scaled_gt_locations = None

        encoder_features, features = self.backbone(image)

        model_output = self.lpn(features, scaled_gt_locations)

        model_output.update(self.detector(
            model_output['lpn_scores'], 
            model_output['lpn_regressions'],
            gt_locations,
            ))

        locations = model_output['training_locations' if self.training else 'pred_locations']
        scaled_locs = locations / jnp.array([height, width])
        instance_output, instance_yc, instance_xc, instance_logits, instance_mask = self.segmentor(features, scaled_locs)
        model_output.update(dict(
            instance_output = instance_output,
            instance_yc = instance_yc,
            instance_xc = instance_xc,
            instance_logtis = instance_logits,
            instance_mask = instance_mask,
        ))

        if self.training and self.auxnet is not None:
            auxnet_input = encoder_features['0'] if self.auxnet.share_weights else image
            op = partial(self._compute_edge, height=height, width=width)
            instance_edge = jax.vmap(op)(instance_output, instance_yc, instance_xc)
            model_output.update(dict(
                auxnet_out = self.auxnet(auxnet_input),
                instance_edge = instance_edge,
            ))

        return model_output
