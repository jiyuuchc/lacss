import tensorflow as tf
from ..ops import *
from .spatial_attention import *
layers = tf.keras.layers

class Segmentor(layers.Layer):
    def __init__(self,
            conv_layers=((64, 64, 64), (16,),),
            instance_crop_size=96,
            feature_scale_ratio=2,
            use_attention=False,
            learned_encoding=False,
            **kwargs,
            ):
        """
          Args:
            conv_layers: conv_block definition
            feature_scale_ratio: 1,2 or 4
            instance_crop_size: crop size, int
            with_attention: T/F whether use spatial attention layer
            learned_encoding: Use hard-coded position encoding or not
        """
        super(Segmentor, self).__init__(**kwargs)

        if feature_scale_ratio != 1 and feature_scale_ratio != 2 and feature_scale_ratio !=4 :
            raise('feature_scale_ratio should be 1,2 or 4')

        self._config_dict = {
            'conv_layers': conv_layers,
            'instance_crop_size': instance_crop_size,
            'feature_scale_ratio': feature_scale_ratio,
            'use_attention': use_attention,
            'learned_encoding': learned_encoding,
        }
        self._config_dict.update(kwargs)

    def get_config(self):
        return self._config_dict

    @property
    def feature_level(self):
        scale = self._config_dict['feature_scale_ratio']
        if scale == 1:
            return 0
        elif scale == 2:
            return 1
        else:
            return 2

    def build(self, input_shape):
        # position encoding
        cs = self._config_dict['instance_crop_size'] // self._config_dict['feature_scale_ratio']
        rr = (tf.range(cs, dtype=tf.float32) - cs/2) / cs
        xx,yy = tf.meshgrid(rr,rr)
        pos = tf.stack([yy,xx], axis=-1)
        pos = pos[None, ...]
        if self._config_dict['learned_encoding']:
            self._pos = pos
        else:
            self._pos = tf.Variable(pos, name='position_encoding')

        #spatail att
        if self._config_dict['use_attention']:
            self._att = SpatialAttention()

        # conv layers
        conv_spec = self._config_dict['conv_layers']
        conv_layers = []
        for k, n_ch in enumerate(conv_spec[0]):
            conv_layers.append(layers.Conv2D(n_ch, 3, name=f'conv_{k}', padding='same', kernel_initializer='he_normal', use_bias=False))
            conv_layers.append(layers.BatchNormalization(name=f'bn_{k}'))
            conv_layers.append(layers.ReLU())
        self._conv_layers = conv_layers

        n_ch = conv_spec[1][0]
        self._feature_conv = layers.Conv2D(n_ch, 1, name='feature_conv', padding='same', kernel_initializer='he_normal', use_bias=False)
        self._code_conv = layers.Conv2D(n_ch, 1, name='code_conv', padding='same', kernel_initializer='he_normal', use_bias=False)
        self._mix_bn = layers.BatchNormalization(name='mix_bn')

        patch_conv_layers = []
        for k, n_ch in enumerate(conv_spec[1][1:]):
            patch_conv_layers.append(layers.Conv2D(n_ch, 1, name=f'patchconv_{k}', padding='same', kernel_initializer='he_normal', use_bias=False))
            patch_conv_layers.append(layers.BatchNormalization(name=f'patch_bn_{k}'))
            patch_conv_layers.append(layers.ReLU())
        self._patch_conv_layers = patch_conv_layers

        if self._config_dict['feature_scale_ratio'] == 1:
            self._output = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name=f'instance_output')
        else:
            self._output = layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid', name=f'instance_output')

        super(Segmentor, self).build(input_shape)

    def call(self, inputs, training=False):
        '''
        Args:
            inputs: (hr_features, locations)
                hr_features: [batch_size, h0, w0, n_ch0]
                locations: [batch_size, None, 2]  ragged tensor, normalzied
            outputs:
                instance_output: [batch_size, None, patch_size, patch_size, 1] ragged tensor, sigmoid
                instance_coords: [batch_size, None, patch_size, patch_size, 2] ragged tensor
        '''
        hr_features, locations = inputs
        crop_size = self._config_dict['instance_crop_size']
        scale = self._config_dict['feature_scale_ratio']
        patch_size = crop_size // scale

        if len(locations.shape) == 2:
            batched_inputs = False
            if len(hr_features.shape) != 3 :
                raise ValueError
            hr_features = tf.expand_dims(hr_features, 0)
        else:
            batched_inputs = True
            locations = locations.to_tensor(-1.0) # convert ragged to padded
            if len(hr_features.shape) != 4 or len(locations.shape) != 3:
                raise ValueError

        if self._config_dict['use_attention']:
            hr_features = self._att(hr_features)

        x = hr_features
        for layer in self._conv_layers:
            x = layer(x, training=training)

        x = self._feature_conv(x)
        if batched_inputs:
            patches, coords = gather_patches(x, locations, patch_size)
            mask = tf.where(locations[:,:,0] >= 0)
            patches = tf.gather_nd(patches, mask)
            coords_y0_x0 = tf.gather_nd(coords[:, :, :1, :1, :], mask)
        else:
            patches, coords = gather_patches(x[0], locations, patch_size)
            coords_y0_x0 = coords[:, :1, :1, :]

        projected_pos = self._code_conv(self._pos)
        patches = patches + projected_pos
        patches = self._mix_bn(patches, training=training)
        patches = tf.nn.relu(patches)

        for layer in self._patch_conv_layers:
            patches = layer(patches, training=training)

        instance_output = self._output(patches, training=training)
        if self._config_dict['feature_scale_ratio'] == 4:
            instance_output = tf.image.resize(instance_output, [crop_size, crop_size])

        # indicies
        rr = tf.range(crop_size, dtype=tf.int32)
        xx,yy = tf.meshgrid(rr, rr)
        mesh = tf.stack([yy,xx], axis=-1)
        instance_coords = coords_y0_x0 * scale + mesh

        if batched_inputs:
            instance_output = tf.RaggedTensor.from_value_rowids(instance_output, mask[:,0])
            instance_coords = tf.RaggedTensor.from_value_rowids(instance_coords, mask[:,0])

        return instance_output, instance_coords
