import tensorflow as tf
import tensorflow.keras.layers as layers
from ..ops import *

class InstanceHead(tf.keras.layers.Layer):
    def __init__(self, n_conv_layers=2, n_conv_channels=64, n_position_encoding_channels=8, instance_crop_size=96, **kwargs):
        self._config_dict = {
            'n_conv_layers': n_conv_layers,
            'n_conv_channels': n_conv_channels,
            'n_position_encoding_channels': n_position_encoding_channels,
            'instance_crop_size': instance_crop_size,
        }
        super(InstanceHead, self).__init__(**kwargs)

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        patch_size = self._config_dict['instance_crop_size']
        n_ch = self._config_dict['n_conv_channels']
        n_position_encoding_channels = self._config_dict['n_position_encoding_channels']

        # self.position_encodings = tf.Variable(
        #       tf.random.normal([patch_size, patch_size, n_position_encoding_channels], dtype=tf.float32),
        #       name='position_encodings'
        #       )

        # hard code position
        cs = self._config_dict['instance_crop_size']
        rr = tf.range(cs, dtype=tf.float32) - cs/2
        xx,yy = tf.meshgrid(rr,rr)
        self.position_encodings = tf.stack([yy,xx,tf.math.abs(yy), tf.math.abs(xx)], axis=-1)

        self.hr_feature_filter = layers.Conv2D(n_ch, 3, name='hr_feature_conv', padding='same', kernel_initializer='he_normal')
        self.lr_feature_filter = layers.Conv2D(n_ch, 3, name='lr_feature_filter', padding='same', kernel_initializer='he_normal', use_bias=False)
        self.pos_filter = layers.Dense(n_ch, name='pos_conv', kernel_initializer='he_normal', use_bias=False)
        self.filter_bn = layers.BatchNormalization(name='filter_bn')

        conv_layers = []
        for k in range(self._config_dict['n_conv_layers']):
            conv_layers.append(layers.ReLU())
            conv_layers.append(layers.Conv2D(n_ch, 3, name=f'conv_{k}', padding='same', kernel_initializer='he_normal'))
            conv_layers.append(layers.BatchNormalization(name=f'bn_{k}'))
        self._conv_layers = conv_layers

        self._output = layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid', name=f'instance_output')


        self._att_conv = layers.Conv2D(1, 7, activation='sigmoid', padding='same', dilation_rate=2)

        super(InstanceHead, self).build(input_shape)

    def call(self, inputs, training=False):
        '''
        Args:
            inputs: (hr_features, lr_features, locations)
                hr_features: [batch_size, h0, w0, n_ch0]
                lr_features: [batch_size, h1, w1, n_ch1]
                locations: [batch_size, N, 2] -1 padded
            outputs:
                instance_output: [batch_size, None, patch_size, patch_size, 1] ragged tensor 0..1
                instance_coords: [batch_size, None, patch_size, patch_size, 2] ragged tensor
        '''
        hr_features, lr_features, locations = inputs
        patch_size = self._config_dict['instance_crop_size']

        if len(locations.shape) == 2:
            batched_inputs = False
            if len(hr_features.shape) != 3 or len(lr_features.shape) != 3:
                raise ValueError
            y0 = self.hr_feature_filter(hr_features[None,...], training=training)[0]
            y1 = self.lr_feature_filter(lr_features[None,...], training=training)[0]
        else:
            batched_inputs = True
            if len(hr_features.shape) != 4 or len(lr_features.shape) != 4 or len(locations.shape) != 3:
                raise ValueError
            y0 = self.hr_feature_filter(hr_features, training=training)
            y1 = self.lr_feature_filter(lr_features, training=taining)

        y0, _ = gather_patches(y0, locations, patch_size)
        y1, _ = gather_patches(y1, locations, 1)
        y2 = self.pos_filter(self.position_encodings, training=training)
        patches = y0 + y1

        #att
        # y_max = tf.reduce_max(y, axis=-1)
        # y_avg = tf.reduce_mean(y, axis=-1)
        # att = self._att_conv(tf.stack([y_max, y_avg], axis=-1), training=training)
        # y = y * att

        # patches = tf.nn.relu(y, name='filter_activation')
        # patches = self.filter_bn(y, training=training)

        if batched_inputs:
            mask = tf.where(locations[:,:,0] >= 0)
            patches = tf.gather_nd(patches, mask)

        for layer in self._conv_layers:
            patches = layer(patches, training=training)
        patches = patches + y2
        patches = tf.nn.relu(patches)

        instance_output = self._output(patches, training=training)

        i_locations = locations * tf.cast(tf.shape(hr_features)[:2], locations.dtype)
        i_locations = tf.cast(i_locations, tf.int32)
        instance_coords = make_meshgrids(i_locations * 2, patch_size * 2)

        if batched_inputs:
            instance_output = tf.RaggedTensor.from_value_rowids(instance_output, mask[:,0])
            instance_coords = tf.RaggedTensor.from_value_rowids(instance_coords, mask[:,0])

        return instance_output, instance_coords
