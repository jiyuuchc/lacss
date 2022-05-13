import tensorflow as tf
import tensorflow.keras.layers as layers

class UNetDownSampler(tf.keras.layers.Layer):
    def __init__(self, filters, n_layers = 2, use_bn=True, kernel_regularizer=None, bias_regularizer=None, **kwargs):
        super(UNetDownSampler, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'filters': filters,
            'n_layers': n_layers,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'use_bn': use_bn,
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
        if use_bn:
            self._norm  = tf.keras.layers.BatchNormalization(name='norm')

    def get_config(self):
        return self._config_dict

    def call(self, inputs, **kwargs):
        x = self._maxpool(inputs)
        for layer in self._conv_block:
            x = layer(x, **kwargs)
        if self._config_dict['use_bn']:
            x = self._norm(x, **kwargs)
        return x

class UNetUpSampler(tf.keras.layers.Layer):
    def __init__(self,
          out_filters,
          n_layers = 2,
          use_bn=True,
          up_conv_method='sample',
          up_conv_filters=-1,
          mix_method='concat',
          kernel_regularizer=None,
          bias_regularizer=None,
          **kwargs):
        super(UNetUpSampler, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'out_filters': out_filters,
            'n_layers': n_layers,
            'use_bn': use_bn,
            'up_conv_method': up_conv_method,
            'up_conv_filters': up_conv_filters,
            'mix_method': mix_method,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
        })

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        x_shape, y_shape = input_shape
        conv_kwargs = {
            'activation': 'relu',
            'padding': 'same',
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': self._config_dict['kernel_regularizer'],
            'bias_regularizer': self._config_dict['bias_regularizer'],
        }

        if self._config_dict['up_conv_method'] == 'sample':
            self._upconv = [layers.UpSampling2D(name='upconv')]

        elif self._config_dict['up_conv_method'] == 'conv':
            if self._config_dict['mix_method'] == 'concat':
                upconv_filters = self._config_dict['up_conv_filters']
                if upconv_filters == -1:
                    upconv_filters = x_shape.as_list()[-1] // 2
            elif self._config_dict['mix_method'] == 'sum':
                upconv_filters = y_shape.as_list()[-1]
            else:
                raise ValueError()

            self._upconv = [
                layers.Conv2DTranspose(upconv_filters, 3, strides=(2,2), name='upconv', **conv_kwargs)
                ]
            if self._config_dict['use_bn']:
                self._upconv.append(layers.BatchNormalization())
        else:
            raise ValueError()

        conv_block = []
        n_filters = self._config_dict['out_filters']
        for i in range(self._config_dict['n_layers']):
            conv_block.append(layers.Conv2D(n_filters, 3, name=f'conv_{i}', **conv_kwargs))
            if self._config_dict['use_bn']:
                conv_block.append(layers.BatchNormalization())
        self._conv_block = conv_block

        super(UNetUpSampler, self).build(input_shape)

    def call(self, inputs, **kwargs):
        ''' inputs = (bottom layer input, encoder input) '''
        x, y = inputs
        for layer in self._upconv:
            x = layer(x)
        if self._config_dict['mix_method'] == 'concat':
            x = tf.concat([x, y], axis=-1)
        else:
            x = x + y
        for layer in self._conv_block:
            x = layer(x, **kwargs)

        return x

#  [n_convs_per_level, (list of channel_nums)]
encoder_configs = {
    's': [2, (64, 128, 256, 512)],
    'n': [2, (128, 256, 512, 1024)]
}

class UNetEncoder(tf.keras.layers.Layer):
    def __init__(self, net_spec,  use_bn=True, **kwargs):
        super(UNetEncoder, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'net_spec': net_spec,
            'use_bn': use_bn,
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
        self._down_stack = []
        for k, ch in enumerate(ch_list, start=1):
            self._down_stack.append(UNetDownSampler(ch, n_layers, use_bn=self._config_dict['use_bn'], name=f'downconv_{k}'),)

        super(UNetEncoder,self).build(input_shape)

    def call(self, data, **kwargs):
        x = data
        # for layer in self._stem:
        #     x=layer(data, **kwargs)
        outputs = [x]
        for k, layer in enumerate(self._down_stack, start=1):
            x = layer(x, **kwargs)
            outputs.append(x)
        return outputs

# (n_convs_per_level, (list of channel_nums in reverse order))
decoder_configs = {
    's': (2, (256, 128, 64, 32)),
    'n': (2, (512, 256, 128, 64)),
}

class UNetDecoder(tf.keras.layers.Layer):
    def __init__(self, net_spec, use_bn=True, up_conv_method='sample', mix_method='concat', **kwargs):
        super(UNetDecoder, self).__init__(**kwargs)
        self._config_dict = kwargs
        self._config_dict.update({
            'net_spec': net_spec,
            'use_bn': use_bn,
            'up_conv_method': up_conv_method,
            'mix_method': mix_method,
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
            layer = UNetUpSampler(
                ch,
                n_layers,
                use_bn=self._config_dict['use_bn'],
                up_conv_method=self._config_dict['up_conv_method'],
                mix_method=self._config_dict['mix_method'],
                name=f'upconv_{lvl}'
                )
            self._up_stack.append(layer)
            lvl -= 1
        super(UNetDecoder, self).build(input_shape)

    def call(self, data, **kwargs):
        x = data[-1]
        outputs = [x]

        for k, layer in enumerate(self._up_stack):
            x = layer((x, data[-k-2]), **kwargs)
            outputs.insert(0, x)

        return outputs

def build_unet_s_backbone(input_shape=(None,None,1)):
    encoder = UNetEncoder((2,[64,128,256,512]), use_bn=False, name='encoder')
    decoder = UNetDecoder((1,[256,128,64]), use_bn=False, up_conv_method='conv', mix_method='sum', name='decoder')
    stem = [
        layers.Conv2D(24,3,activation='relu', padding='same', name='stem_conv1'),
        layers.Conv2D(32,3,activation='relu', padding='same', name='stem_conv2'),
        # layers.BatchNormalization(name='stem_bn'),
    ]
    input = Layers.Input(shape=input_shape)
    y = input
    for layer in stem:
        y = layer(y)
    y = encoder(y)
    y = decoder(y)
    output = (y[0], y[2])
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

def build_unet_l_backbone(input_shape=(None,None,1)):
    encoder = UNetEncoder((2,[128,256,512,1024]), use_bn=False, name='encoder')
    decoder = UNetDecoder((1,[512,256,128]), use_bn=False, up_conv_method='conv', mix_method='sum', name='decoder')
    stem = [
        layers.Conv2D(32,3,activation='relu', padding='same', name='stem_conv1'),
        layers.Conv2D(64,3,activation='relu', padding='same', name='stem_conv2'),
        # layers.BatchNormalization(name='stem_bn'),
    ]
    input = Layers.Input(shape=input_shape)
    y = input
    for layer in stem:
        y = layer(y)
    y = encoder(y)
    y = decoder(y)
    output = (y[0], y[2])
    model = tf.keras.Model(inputs=input, outputs=output)
    return model
