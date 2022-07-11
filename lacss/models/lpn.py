import tensorflow as tf
from .channel_attention import ChannelAttention
layers = tf.keras.layers

class LPN(tf.keras.layers.Layer):
    def __init__(self,
          conv_layers=((1024,),(1024,)),
          with_channel_attention=False,
          activation='relu',
          **kwargs,
          ):
        """
          Args:
            conv_layers: conv layer spec
            with_channel_attention: whether include channel attention
        """
        super(LPN, self).__init__(**kwargs)
        self._config_dict = {
            'conv_layers': conv_layers,
            'with_channel_attention': with_channel_attention,
            'activation': activation,
        }
        self._config_dict.update(kwargs)

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        conv_kwargs = {
            'padding': 'same',
            'kernel_initializer': 'he_normal',
            'use_bias': False,
        }

        conv_layer_spec = self._config_dict['conv_layers']
        activation = self._config_dict['activation']
        conv_layers = []
        for k, n_ch in enumerate(conv_layer_spec[0]):
            conv_layers.append(layers.Conv2D(n_ch, 3, name=f'conv3_{k}', **conv_kwargs))
            conv_layers.append(layers.BatchNormalization(name=f'conv_bn_{k}'))
            conv_layers.append(layers.Activation(activation))
        for k, n_ch in enumerate(conv_layer_spec[1]):
            conv_layers.append(layers.Conv2D(n_ch, 1, name=f'conv1_{k}', **conv_kwargs))
            conv_layers.append(layers.BatchNormalization(name=f'conv1_bn_{k}'))
            conv_layers.append(layers.Activation(activation))
        if self._config_dict['with_channel_attention']:
            conv_layers.append(ChannelAttention())

        self._conv_layers = conv_layers

        self._score_layer = layers.Dense(1, name='score', activation='sigmoid', kernel_initializer='he_normal')
        self._regression_layer = layers.Dense(2, name='regression', kernel_initializer='he_normal')

        super(LPN, self).build(input_shape)

    def call(self, inputs, training=False):
        '''
        Args:
            inputs: [batch_size, H, W, ch]
        Returns:
            scores: [batch_size, H, W, 1]
            regression: [batch_size, H, W, 2]
        '''

        input_shape = tf.shape(inputs)

        y = inputs

        batched_input = True
        if len(input_shape) == 3:
            y = tf.expand_dims(y, 0)
            batched_input = False

        for layer in self._conv_layers:
            y = layer(y, training=training)

        scores_out = self._score_layer(y, training=training)
        regression_out = self._regression_layer(y, training=training)

        if not batched_input:
            scores_out = scores_out[0]
            regression_out = regression_out[0]

        return scores_out, regression_out
