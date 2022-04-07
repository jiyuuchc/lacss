import tensorflow as tf
import tensorflow.keras.layers as layers

class UNetDownSampler(tf.keras.layers.Layer):
    def __init__(self, filters, n_layers = 2, kernel_regularizer=None, bias_regularizer=None, **kwargs):
        super(UNetDownSampler, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'filters': filters,
            'n_layers': n_layers,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
        })
        conv_kwargs = {
            'activation': 'relu',
            'padding': 'same',
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
        }
        self._maxpool = tf.keras.layers.MaxPool2D(name='maxpool')
        conv_block = []
        for i in range(n_layers):
            conv_block.append(layers.Conv2D(filters, 3, name=f'conv_{i}', **conv_kwargs))
        self._conv_block = conv_block
        self._norm  = tf.keras.layers.BatchNormalization(name='norm')

    def get_config(self):
        return self._config_dict

    def call(self, inputs, **kwargs):
        x = self._maxpool(inputs)
        for layer in self._conv_block:
            x = layer(x, **kwargs)
        x = self._norm(x, **kwargs)
        return x

class UNetUpSampler(tf.keras.layers.Layer):
    def __init__(self, filters, n_layers = 2, kernel_regularizer=None, bias_regularizer=None, **kwargs):
        super(UNetUpSampler, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'filters': filters,
            'n_layers': n_layers,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
        })
        conv_kwargs = {
            'activation': 'relu',
            'padding': 'same',
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
        }
        self._upconv = layers.UpSampling2D(name='upsampling')
        conv_block = []
        for i in range(n_layers):
            conv_block.append(layers.Conv2D(filters, 3, name=f'conv_{i}', **conv_kwargs))
        self._conv_block = conv_block
        self._norm  = tf.keras.layers.BatchNormalization(name='norm')

    def get_config(self):
        return self._config_dict

    def call(self, inputs, **kwargs):
        ''' inputs = (bottom layer input, encoder input) '''
        x, y = inputs
        x = self._upconv(x)
        x = tf.concat([x, y], axis=-1)
        for layer in self._conv_block:
            x = layer(x, **kwargs)
        x = self._norm(x, **kwargs)
        return x

#  [n_convs_per_level, (list of channel_nums)]
encoder_configs = {
    'a': [2, (32, 64, 128, 256)],
    'b': [2, (48, 96, 192, 384)],
}

class UNetEncoder(tf.keras.layers.Layer):
    def __init__(self, net_spec,  **kwargs):
        super(UNetEncoder, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'net_spec': net_spec,
        })
        self.net_config = encoder_configs[net_spec]

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        n_layers = self.net_config[0]
        n_stem_filters = self.net_config[1]
        ch_list = self.net_config[2]
        # self._stem = [
        #       tf.keras.layers.Conv2D(n_stem_filters, 3, name='stem', padding='same', activation='relu', kernel_initializer='he_normal'),
        #       tf.keras.layers.BatchNormalization(name='norm'),
        #       ]
        self._down_stack = []
        for k, ch in enumerate(ch_list, start=1):
            self._down_stack.append(UNetDownSampler(ch, n_layers, name=f'downconv_{k}'),)

        super(UNetEncoder,self).build(input_shape)

    def call(self, data, **kwargs):
        x = data
        for layer in self._stem:
            x=layer(data, **kwargs)
        outputs = [x]
        for k, layer in enumerate(self._down_stack, start=1):
            x = layer(x, **kwargs)
            outputs.append(x)
        return outputs

# (n_convs_per_level, (list of channel_nums in reverse order))
decoder_configs = {
    'a': (2, (128, 64, 32, 16)),
    'b': (2, (192, 96, 48, 24)),
}

class UNetDecoder(tf.keras.layers.Layer):
    def __init__(self, net_spec, **kwargs):
        super(UNetDecoder, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'net_spec': net_spec,
        })
        if isinstance(net_spec, str):
          self.net_config = decoder_configs[net_spec]
        else:
          self.net_config = net_spec

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        n_layers = self.net_config[0]
        ch_list = self.net_config[1]
        max_level = len(ch_list)

        self._up_stack = []
        lvl =  max_level - 1
        for ch in ch_list:
            self._up_stack.append(UNetUpSampler(ch, n_layers, name=f'upconv_{lvl}'))
            lvl -= 1
        super(UNetDecoder, self).build(input_shape)

    def call(self, data, **kwargs):
        x = data[-1]
        outputs = [x]

        for k, layer in enumerate(self._up_stack):
            x = layer((x, data[-k-2]), **kwargs)
            outputs.insert(0, x)

        return outputs
