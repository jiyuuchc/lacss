import tensorflow as tf
from .channel_attention import ChannelAttention

layers = tf.keras.layers

class Bottleneck(layers.Layer):
    def __init__(self,
                 filters,
                 strides=1,
                 dilation_rate=1,
                 use_attention=True,
                 **kwargs):
      super(Bottleneck, self).__init__(**kwargs)
      self.filters = filters
      self.strides = strides
      self.dilation_rate = dilation_rate
      self.use_attention = use_attention

    def build(self, input_shape):
        n_filters = self.filters
        strides = self.strides
        dilation_rate= self.dilation_rate
        if strides > 1:
            self._shortcut_layers = [
                layers.Conv2D(n_filters * 4, 1, strides=strides, use_bias=False, kernel_initializer='he_normal'),
                layers.BatchNormalization(),
                ]
        else:
            self._shortcut_layers = []

        self._conv_layers = [
            layers.Conv2D(n_filters, 1, use_bias=False, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(
                n_filters, 3,
                strides=strides,
                dilation_rate=dilation_rate,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n_filters * 4, 1, use_bias=False, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU(),
            ]

        if self.use_attention:
            self._conv_layers.append(ChannelAttention())

        super(Bottleneck, self).build(input_shape)

    def call(self, inputs, training=None):
        shortcut = inputs
        for layer in self._shortcut_layers:
            shortcut = layer(shortcut)

        x = inputs
        for layer in self._conv_layers:
            x = layer(x)

        x = x + shortcut
        return tf.nn.relu(x)

class FPN(layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape):
        conv_kwargs = {
            'padding': 'same',
            'activation': 'relu',
            'kernel_initializer': 'he_normal',
        }

        self._fpn_convs = []
        for k in range(len(input_shape)):
            self._fpn_convs.append(layers.Conv2D(self.out_channels, 3, **conv_kwargs))

        self._out_convs = []
        self._upsamples = []
        for k in range(len(input_shape)-1):
            self._out_convs.append(layers.Conv2D(self.out_channels, 3, **conv_kwargs))
            self._upsamples.append(layers.UpSampling2D())

        super().build(input_shape)

    def call(self, inputs, training=None):
        output = [op(x) for x, op in zip(inputs, self._fpn_convs)]
        for k in range(len(output)-1, 0, -1):
            x = self._upsamples[k-1](output[k]) + output[k-1]
            x = self._out_convs[k-1](x, training=training)
            output[k-1] = x

        return output

_RESNET_SPECS = {
  '50': [(64,3),(128,4),(256,6),(512,3)],
  '101': [(64,3),(128,4),(256,23),(512,3)],
  '152': [(64,3),(128,8),(256,36),(512,3)],
  '200': [(64,3),(128,24),(256,36),(512,3)],
  '270': [(64,4),(128,29),(256,53),(512,4)],
  '350': [(64,4),(128,36),(256,72),(512,4)],
  '420': [(64,4),(128,44),(256,87),(512,4)],
}

class ResNet(layers.Layer):
    def __init__(self, model_spec, use_attention=True, min_feature_level=1, out_channels=512, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self._config_dict = {
            'model_spec': model_spec,
            'use_attention': use_attention,
            'min_feature_level': min_feature_level,
            'out_channels': out_channels,
        }
        self._config_dict.update(kwargs)

    def build(self, input_shape):
        spec = self._config_dict['model_spec']
        if isinstance(spec, str):
            spec = _RESNET_SPECS[spec]
        use_attention = self._config_dict['use_attention']

        self._stem = [
            layers.Conv2D(24, 3, padding='same', activation='relu',name='stem_conv1', kernel_initializer='he_normal'),
            layers.Conv2D(64, 3, padding='same', activation='relu',name='stem_conv2', kernel_initializer='he_normal'),
            layers.BatchNormalization(name='stem_bn'),
        ]

        all_blocks = []
        for n_channels, n_repeats in spec:
            block = []
            block.append(Bottleneck(n_channels, 2, use_attention=use_attention))
            for k in range(1, n_repeats):
                block.append(Bottleneck(n_channels, use_attention=use_attention))
            all_blocks.append(block)

        self._encoder_layers = all_blocks

        self._fpn = FPN(self._config_dict['out_channels'])

        super(ResNet, self).build(input_shape)

    def call(self, x, training=None):
        encoder_out = []
        for op in self._stem:
            x = op(x, training=training)
        encoder_out.append(x)

        for block in self._encoder_layers:
            for op in block:
                x = op(x)
            encoder_out.append(x)

        l_min = self._config_dict['min_feature_level']
        decoder_out = self._fpn(encoder_out[l_min:], training=training)

        keys = [str(k) for k in range(l_min, len(decoder_out) + l_min)]
        decoder_out = dict(zip(keys, decoder_out))

        return decoder_out

    def get_config(self):
        return self._config_dict


# def build_resnet_backbone(input_shape=(None,None,1), is_v2=False, with_attention=True, spec='50'):
#     encoder = ResNet(_RESNET_SPECS[spec], use_attention=with_attention, name='resnet')
#     input = tf.keras.layers.Input(shape=input_shape)
#     x = input
#     x = layers.Conv2D(24,3,padding='same', activation='relu',name='stem_conv1', kernel_initializer='he_normal')(x)
#     if is_v2:
#         x0 = x
#         x = layers.Conv2D(64,3,strides=2,padding='same', activation='relu',name='stem_conv2', kernel_initializer='he_normal')(x0)
#         x = layers.BatchNormalization(name='stem_bn')(x)
#         stem = (x0, x)
#     else:
#         x = layers.Conv2D(64,3,padding='same', activation='relu',name='stem_conv2', kernel_initializer='he_normal')(x)
#         x = layers.BatchNormalization(name='stem_bn')(x)
#         stem = (x,)
#
#     x = encoder(x)
#     encoder_out = stem + tuple(x)
#
#     x = [layers.Conv2D(256,1,activation='relu',name=f'fpn_conv{k}_1')(d) for k,d in enumerate(x)]
#     out5 = x[-1]
#
#     m4 = layers.UpSampling2D(name='upsample_4')(out5) + x[-2]
#     out4 = layers.Conv2D(256, 3, activation='relu', padding='same', name=f'fpn_conv4_2')(m4)
#
#     m3 = layers.UpSampling2D(name='upsample_3')(out4) + x[-3]
#     out3 = layers.Conv2D(256, 3, activation='relu', padding='same', name=f'fpn_conv3_2')(m3)
#
#     m2 = layers.UpSampling2D(name='upsample_2')(out3) + x[-4]
#     out2 = layers.Conv2D(256, 3, activation='relu', padding='same', name=f'fpn_conv2_2')(m2)
#
#     decoder_out = (out2, out3, out4, out5)
#     if is_v2:
#         m1 = tf.concat([layers.UpSampling2D(name='upsample_1')(out2), encoder_out[1]], axis=-1)
#         out1 = layers.Conv2D(256, 3, activation='relu', padding='same', name=f'fpn_conv1_2')(m1)
#         decoder_out = (out1,) + decoder_out
#
#     return tf.keras.Model(inputs=input, outputs=(encoder_out, decoder_out))
