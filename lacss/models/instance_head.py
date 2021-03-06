import tensorflow as tf
import tensorflow.keras.layers as layers
from ..ops import *

class InstanceHead(tf.keras.layers.Layer):
    def __init__(self, n_conv_layers=3, n_conv_channels=64, n_patch_conv_layers=1, instance_crop_size=96, with_attention=False, **kwargs):
        self._config_dict = {
            'n_conv_layers': n_conv_layers,
            'n_conv_channels': n_conv_channels,
            'n_patch_conv_layers': n_patch_conv_layers,
            'instance_crop_size': instance_crop_size,
        }
        super(InstanceHead, self).__init__(**kwargs)

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        n_ch = self._config_dict['n_conv_channels']

        # hard code position
        cs = self._config_dict['instance_crop_size']
        rr = (tf.range(cs, dtype=tf.float32) - cs/2)/cs
        xx,yy = tf.meshgrid(rr,rr)
        self._position_encodings = tf.stack([yy,xx], axis=-1)
        # self._position_encodings = tf.stack([yy,xx,tf.math.abs(yy), tf.math.abs(xx)], axis=-1)
        self._position_encodings = tf.expand_dims(self._position_encodings, 0)

        conv_layers = []
        for k in range(self._config_dict['n_conv_layers']):
            conv_layers.append(layers.Conv2D(n_ch, 3, name=f'conv_{k}', padding='same', kernel_initializer='he_normal'))
            conv_layers.append(layers.BatchNormalization(name=f'bn_{k}'))
            conv_layers.append(layers.ReLU())
        self._conv_layers = conv_layers

        self._feature_conv = layers.Conv2D(n_ch//4, 1, name='feature_conv', padding='same', kernel_initializer='he_normal')
        self._code_conv = layers.Conv2D(n_ch//4, 1, name='code_conv', padding='same', kernel_initializer='he_normal', use_bias=False)
        # self._code_conv = layers.Conv2D(8, 1, name='code_conv', padding='same', kernel_initializer='he_normal', use_bias=False)
        self._mix_bn = layers.BatchNormalization(name='mix_bn')

        patch_conv_layers = []
        for k in range(self._config_dict['n_patch_conv_layers']-1):
            patch_conv_layers.append(layers.Conv2D(n_ch//4, 1, name=f'patchconv_{k}', padding='same', kernel_initializer='he_normal'))
            patch_conv_layers.append(layers.BatchNormalization(name=f'patch_bn_{k}'))
            patch_conv_layers.append(layers.ReLU())
        self._patch_conv_layers = patch_conv_layers

        self._output = layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid', name=f'instance_output')

        super(InstanceHead, self).build(input_shape)

    def call(self, inputs, training=False):
        '''
        Args:
            inputs: (hr_features, locations)
                hr_features: [batch_size, h0, w0, n_ch0]
                locations: [batch_size, N, 2] -1 padded
            outputs:
                instance_output: [batch_size, None, patch_size, patch_size, 1] ragged tensor 0..1
                instance_coords: [batch_size, None, patch_size, patch_size, 2] ragged tensor
        '''
        hr_features, locations = inputs
        patch_size = self._config_dict['instance_crop_size']

        if len(locations.shape) == 2:
            batched_inputs = False
            if len(hr_features.shape) != 3 :
                raise ValueError
            hr_features = tf.expand_dims(hr_features, 0)
        else:
            batched_inputs = True
            if len(hr_features.shape) != 4 or len(locations.shape) != 3:
                raise ValueError

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

        n_ch = self._config_dict['n_conv_channels']
        x1 = self._code_conv(self._position_encodings)
        # x1 = tf.pad(x1, [[0,0],[0,0],[0,0],[n_ch-n_ch//4, 0]])
        patches = patches + x1
        patches = self._mix_bn(patches, training=training)
        patches = tf.nn.relu(patches)

        # n_patches = tf.shape(patches)[0]
        # encodings = tf.tile(self._position_encodings, [n_patches, 1, 1, 1])
        # patches = tf.concat([patches, encodings], axis=-1)
        for layer in self._patch_conv_layers:
            patches = layer(patches, training=training)

        instance_output = self._output(patches, training=training)

        # i_locations = locations * tf.cast(tf.shape(hr_features)[-3:-1], locations.dtype)
        # i_locations = tf.cast(i_locations, tf.int32)
        # instance_coords = make_meshgrids(i_locations * 2, patch_size * 2)
        rr = tf.range(patch_size * 2, dtype=tf.int32)
        xx,yy = tf.meshgrid(rr, rr)
        mesh = tf.stack([yy,xx], axis=-1)
        instance_coords = coords_y0_x0 * 2 + mesh

        if batched_inputs:
            instance_output = tf.RaggedTensor.from_value_rowids(instance_output, mask[:,0])
            instance_coords = tf.RaggedTensor.from_value_rowids(instance_coords, mask[:,0])

        return instance_output, instance_coords
