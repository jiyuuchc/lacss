import tensorflow as tf
layers = tf.keras.layers

class SpatialAttention(layers.Layer):
    def __init__(self, filter_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self._w = layers.Conv2D(1, self._config_dict['filter_size'], activation='sigmoid')
        self.filter_size = filter_size

    def call(self, x, training=False):
        y1 = tf.reduce_max(x, axis=-1)
        y2 = tf.reduce_mean(x, axis=-1)
        y = tf.stack([y1,y2], axis=-1)
        y = self._w(y)

        y = x * y

        return y
