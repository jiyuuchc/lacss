import tensorflow as tf
layers = tf.keras.layers

_unet_configs = {
    'unet_s': (32, 64, 128, 256, 512),
    'unet_n': (64, 128, 256, 512, 1024),
    'unet_l': (96, 192, 384, 768, 1536),
}

class MixingBlock(layers.Layer):
    def __init__(self, method='upsampling', hidden_dim=None, out_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

    def build(self, input_shape):
        conv_kwargs = {
            'padding': 'same',
            'kernel_initializer': 'he_normal',
        }

        if self.method == 'conv_add' or self.hidden_dim is None:
            n_channels = input_shape[0][-1]
        else:
            n_channels = self.hidden_dim
        if self.method == 'conv_add' or self.method == 'conv_concat':
            self._upconv = layers.Conv2DTranspose(n_channels, 3, strides=2, **conv_kwargs)
        elif self.method == 'upsampling':
            self._upconv = layers.UpSampling2D()
        else:
            raise ValueError('Valid methods are conv_add|conv_concat|upsampling')

    def call(self, inputs, training=None):
        x, y = inputs
        y = self._upconv(y, training=training)
        if self.method == 'conv_add':
            x = x + y
        else:
            x = tf.concat((x,y), axis=-1)

        return x

class UNet(layers.Layer):
    def __init__(self, net_spec='unet_n', use_bn=False, method='conv_concat', min_feature_level=1, activation='relu', **kwargs):
        super(UNet, self).__init__(**kwargs)
        self._config_dict = {
            'net_spec': net_spec,
            'use_bn': use_bn,
            'method': method,
            'min_feature_level': min_feature_level,
            'activation': activation,
        }
        self._config_dict.update(kwargs)

    def get_config(self):
        return self._config_dict

    def build_conv_block(self, out_channels, repeats=2):
        conv_kwargs = {
            'padding': 'same',
            'kernel_initializer': 'he_normal',
            'use_bias': not self._config_dict['use_bn'],
        }
        block = []
        for k in range(repeats):
            block.append(layers.Conv2D(out_channels, 3, **conv_kwargs))
            if self._config_dict['use_bn']:
                block.append(layers.BatchNormalization())
            block.append(layers.Activation(self._config_dict['activation']))

        return block

    def build(self, input_shape):
        net_spec = self._config_dict['net_spec']
        if isinstance(net_spec, str):
            net_spec = _unet_configs[net_spec]
        n_levels = len(net_spec)

        self._encoder_conv_list = []
        for k, n_channels in enumerate(net_spec):
            self._encoder_conv_list.append(
                self.build_conv_block(n_channels)
            )

        l_min = self._config_dict['min_feature_level']
        self._decoder_conv_list = []
        self._mixer_list = []
        for n_channels in net_spec[l_min:-1]:
            self._decoder_conv_list.append(self.build_conv_block(n_channels))
            self._mixer_list.append(MixingBlock(method=self._config_dict['method']))

        super(UNet, self).build(input_shape)

    def call(self, x, training=None):
        output = []

        for block in self._encoder_conv_list:
            for layer in block:
                x = layer(x, training=training)
            output.append(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        l_min = self._config_dict['min_feature_level']
        output = output[l_min:]
        for k in range(len(output)-1, 0, -1):
            mixer = self._mixer_list[k-1]
            conv_block = self._decoder_conv_list[k-1]
            x = mixer((output[k-1], output[k]), training=training)
            for layer in conv_block:
                x = layer(x, training=training)
            output[k-1] = x

        keys = [str(k) for k in range(l_min, len(output)+l_min)]
        output = dict(zip(keys, output))

        return output
