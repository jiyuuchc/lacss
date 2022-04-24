import tensorflow as tf
import tensorflow.keras.layers as layers
from ..ops import *

class InstanceHead(tf.keras.layers.Layer):
    def __init__(self, n_conv_layers=1, n_conv_channels=32, n_att_channels=4, instance_crop_size=96, **kwargs):
        self._config_dict = {
            'n_conv_layers': n_conv_layers,
            'n_conv_channels': n_conv_channels,
            'n_att_channels': 4,
            'instance_crop_size': instance_crop_size,
        }
        super(InstanceHead, self).__init__(**kwargs)

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        conv_kwargs = {
            'padding': 'same',
            'kernel_initializer': 'he_normal',
        }

        patch_size = self._config_dict['instance_crop_size']
        delta = 2.0 / (patch_size - 1)
        s = tf.range(-1., 1 + delta/2, delta)
        xc, yc = tf.meshgrid(s, s)
        encodings = tf.stack([yc, xc, yc**2, xc**2, yc**3, xc**3], axis=-1)
        self._pos_encoding = tf.expand_dims(encodings, 0)

        conv_layers = []
        n_ch = self._config_dict['n_conv_channels']
        for k in range(self._config_dict['n_conv_layers']):
            conv_layers.append(layers.Conv2D(n_ch, 3, name=f'conv_{k}', activation='relu', **conv_kwargs))
            conv_layers.append(layers.BatchNormalization(name=f'bn_{k}'))
        conv_layers.append(layers.Conv2D(self._config_dict['n_att_channels'], 1, name=f'conv_last', activation='relu', **conv_kwargs))
        conv_layers.append(layers.BatchNormalization(name=f'bn_last'))
        self._conv_layers = conv_layers

        self._att_layer_a = layers.Dense(self._config_dict['n_att_channels'], name='att_a')
        self._att_layer_b = layers.Conv2D(self._config_dict['n_att_channels'], 1, name='att_b', **conv_kwargs)
        self._output = layers.Conv2D(1, 1, name=f'conv_last', activation='sigmoid', **conv_kwargs)

        super(InstanceHead, self).build(input_shape)

    def call(self, inputs, training=False):
        '''
        Args:
            inputs: (hr_features, lr_features, locations)
                hr_features: [batch_size, h0, w0, n_ch0]
                lr_features: [batch_size, h1, w1, n_ch1]
                locations: [batch_size, N, 2] -1 padded
            outputs:
                instance_output: [batch_size, None, patch_size, patch_size, 1] ragged tensor
                instance_coords: [batch_size, None, patch_size, patch_size, 2] ragged tensor
        '''
        hr_features, lr_features, locations = inputs
        patch_size = self._config_dict['instance_crop_size']

        x = hr_features
        for layer in self._conv_layers:
            x = layer(x, training=training)

        mask = tf.where(locations[:,:,0] >= 0)
        locations = tf.clip_by_value(locations, 0, 1)

        patches, coords = gather_patches(x, locations, self._config_dict['instance_crop_size'])
        patches = tf.gather_nd(patches, mask)
        coords = tf.gather_nd(coords, mask)

        lr_locations = locations * tf.cast(tf.shape(lr_features)[1:3], locations.dtype)
        lr_locations = tf.cast(lr_locations + .5, tf.int32)
        center_features = tf.gather_nd(lr_features, lr_locations, batch_dims=1)
        center_features = tf.gather_nd(center_features, mask)

        att_a = self._att_layer_a(center_features)
        att_a = att_a[:, None, None, :]

        att_b = self._att_layer_b(self._pos_encoding)

        att = tf.sigmoid(att_a + att_b) #broadcast
        patches = patches * att

        instance_output = self._output(patches, training=training)
        instance_output = tf.RaggedTensor.from_value_rowids(instance_output, mask[:,0])

        instance_coords = tf.RaggedTensor.from_value_rowids(coords, mask[:,0])

        return instance_output, instance_coords
