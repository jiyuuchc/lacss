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
        self._w1 = layers.Dense(n_ch, name='w1')

    def call(self, x, training=False):
        n_batch = tf.shape(x)[0]
        y = tf.reshape(x, (n_batch, -1, x.shape[-1]))
        y1 = tf.reduce_max(y, axis=1)
        y1 = self._w0(y1)
        y2 = tf.reduce_mean(y, axis=1)
        y2 = self._w0(y2)
        y = self._w1(y1 + y2)
        y = tf.sigmoid(y)
        y = x * y[:, None, None, :]

        return y
