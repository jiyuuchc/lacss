import tensorflow as tf
import tensorflow.keras.layers as layers

class DetectionHead(tf.keras.layers.Layer):
    def __init__(self, conv_filters=(1024,), fc_filters=(1024,), **kwargs):
        self._config_dict = {
            'conv_filters': conv_filters,
            'fc_filters': fc_filters,
        }
        super(DetectionHead, self).__init__(**kwargs)

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        conv_kwargs = {
            'padding': 'same',
            'activation': 'relu',
            'kernel_initializer': 'he_normal',
        }
        conv_filters = self._config_dict['conv_filters']
        fc_filters = self._config_dict['fc_filters']
        conv_layers = []
        for k in range(len(conv_filters)):
            conv_layers.append(layers.Conv2D(conv_filters[k], 3, name=f'conv_{k}', **conv_kwargs))
            conv_layers.append(layers.BatchNormalization(name=f'conv_bn_{k}'))
        for k in range(len(fc_filters)):
            conv_layers.append(layers.Conv2D(fc_filters[k], 1, name=f'fc_{k}', **conv_kwargs))
            conv_layers.append(layers.BatchNormalization(name=f'fc_bn_{k}'))
        self._conv_layers = conv_layers

        self._score_layer = layers.Dense(1, name='score', activation='sigmoid', kernel_initializer='he_normal')
        self._regression_layer = layers.Dense(2, name='regression', kernel_initializer='he_normal')

        super(DetectionHead, self).build(input_shape)

    def call(self, inputs, training=False):
        '''
        Args:
            inputs: [batch_size, H, W, ch]
        Returns:
            scores: [batch_size, H, W, 1]
        '''

        input_shape = tf.shape(inputs)

        y = inputs

        for layer in self._conv_layers:
            y = layer(y)

        scores_out = self._score_layer(y)
        regression_out = self._regression_layer(y)

        return scores_out, regression_out
