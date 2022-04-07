import tensorflow as tf
import tensorflow.keras.layers as layers

''' Shufflenet V2 backbone
    This impementation does not include the final 1x1 conv and globalpool layer.
    Constructor takes one string input, indicating the subtype {0.5x|1x|1.5x|2x}
    Input to the call function should be reshaped to size that is multiple of 32.
    Call function output a dict with string keys ("2"-"5") indicating the layers.
'''

def _shuffle_xy(xy):
    _, height, width, channels = xy.get_shape()
    xy_split = tf.stack(tf.split(xy, num_or_size_splits=2, axis=-1), axis=-1)
    return tf.reshape(xy_split, [-1, height, width, channels])

class ShuffleDownUnit(tf.keras.layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super(ShuffleDownUnit, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'n_channels': n_channels,
        })
        self._block_y = [
            layers.Conv2D(n_channels//2, 1, kernel_initializer='he_normal', name='cy1'),
            layers.BatchNormalization(name='cy1_norm'),
            layers.ReLU(),
            layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_normal', name='cy2'),
            layers.BatchNormalization(name='cy2_norm'),
            layers.Conv2D(n_channels//2, 1, kernel_initializer='he_normal', name='cy3'),
            layers.BatchNormalization(name='cy3_norm'),
            layers.ReLU(),
        ]
        self._block_x = [
            layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='SAME', kernel_initializer='he_normal', name='cx1'),
            layers.BatchNormalization(name='cx1_norm'),
            layers.Conv2D(n_channels//2, 1, kernel_initializer='he_normal', name='cx2'),
            layers.BatchNormalization(name='cx2_norm'),
            layers.ReLU(),
        ]

    def get_config(self):
        return self._config_dict

    def call(self, xy, **kwargs):
        y = xy
        for layer in self._block_y:
            y = layer(y)
        x = xy
        for layer in self._block_x:
            x = layer(x)
        xy = tf.concat([x,y], axis=-1)
        return _shuffle_xy(xy)

class ShuffleUnit(tf.keras.layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super(ShuffleUnit, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'n_channels': n_channels,
        })

        self._block = [
            layers.Conv2D(n_channels//2, 1, kernel_initializer='he_normal', name='c1'),
            layers.BatchNormalization(name='c1_norm'),
            layers.ReLU(),
            layers.DepthwiseConv2D(kernel_size=3, padding='SAME', kernel_initializer='he_normal', name='c2_conv'),
            layers.BatchNormalization(name='c2_norm'),
            layers.Conv2D(n_channels//2, 1, kernel_initializer='he_normal', name='c3'),
            layers.BatchNormalization(name='c3_norm'),
            layers.ReLU(),
        ]

    def get_config(self):
        return self._config_dict

    def call(self, xy, **kwargs):
        x, y = tf.split(xy, num_or_size_splits=2, axis=-1)
        for layer in self._block:
            y = layer(y, **kwargs)
        xy = tf.concat([x,y], axis=-1)
        return _shuffle_xy(xy)

class ShuffleBlock(tf.keras.layers.Layer):
    def __init__(self, n_units, n_channels, **kwargs):
        super(ShuffleBlock, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'n_units': n_units,
            'n_channels': n_channels,
        })

        self._stack = [ShuffleDownUnit(n_channels, name='down')]
        for k in range(n_units):
            self._stack.append(ShuffleUnit(n_channels, name=f'shuffle_{k+1}'))

    def get_config(self):
        return self._config_dict

    def call(self, xy, **kwargs):
        for layer in self._stack:
            xy = layer(xy)
        return xy

shuffle_net_configs = {
    '0.5x': (24, [(3, 48), (7, 96), (3, 192)]),
    '1x': (24, [(3, 116), (7, 232), (3, 464)]),
    '1.5x': (24, [(3, 176), (7, 352), (3, 704)]),
    '2x': (24, [(3, 244), (7, 488), (3, 976)]),
}

class ShuffleNetV2(tf.keras.layers.Layer):
    def __init__(self, config_key, with_stem=True, **kwargs):
        super(ShuffleNetV2, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'config_key': config_key,
            'with_stem': with_stem,
        })
        net_configs = shuffle_net_configs[config_key]
        stem_channels = net_configs[0]

        self._stem = [
          layers.Conv2D(stem_channels, 3, strides = 2, padding='same', name='stem_conv'),
          layers.BatchNormalization(name='stem_norm'),
          layers.MaxPool2D(name='stem_maxpool', pool_size=(3,3), strides=(2,2), padding='same'),
        ]

        blocks = []
        for n_units, n_channels in net_configs[1]:
            blocks.append(ShuffleBlock(n_units=n_units, n_channels=n_channels))
        self._blocks = blocks

    def get_config(self):
        return self._config_dict

    def call(self, x, **kwargs):
        if self._config_dict['with_stem']:
            for layer in self._stem:
                  x = layer(x)
            outputs = [x]
        else:
            outputs = [x]

        for k, shuffle_block in enumerate(self._blocks):
            x = shuffle_block(x)
            outputs.append(x)
        return outputs
