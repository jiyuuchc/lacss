import tensorflow as tf
import tensorflow.keras.layers as layers

class ChannelAttention(layers.Layer):
    def __init__(self, squeeze_factor=16, **kwargs):
        self._config_dict = {
            'squeeze_factor': squeeze_factor,
        }
        super(ChannelAttention, self).__init__(**kwargs)

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        n_ch = input_shape[-1]
        self._w0 = layers.Dense(
            n_ch//self._config_dict['squeeze_factor'],
            activation='relu',
            name='w0')
        self._w1 = layers.Dense(n_ch, name='relu')

    def call(self, x, training=False):
        y = tf.reshape(x, (-1, x.shape[-1]))
        y1 = tf.reduce_max(y, axis=0, keepdims=True)
        y1 = self._w0(y1)
        y2 = tf.reduce_mean(y, axis=0, keepdims=True)
        y2 = self._w0(y2)
        y = self._w1(y1 + y2)
        y = tf.sigmoid(y)
        y = x * y

        return y
