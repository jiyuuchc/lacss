import math
import tensorflow as tf
from .unet import MixingBlock

layers = tf.keras.layers

''' XCiT model
    based on https://github.com/facebookresearch/xcit
    ref: https://arxiv.org/abs/2106.09681
'''

class PositionalEncodingFourier(layers.Layer):
    ''' positonal encoding as the original tansformer model
    '''

    def __init__(self, hidden_dim=32, temperature=10000, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        dim_t = tf.range(hidden_dim, dtype=tf.float32) // 2 * 2 / hidden_dim
        dim_t = temperature ** dim_t
        self._dim_t = dim_t

    def build(self, input_shape):
        self._token_projection = layers.Dense(input_shape[-1])
        super().build(input_shape)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        yy = tf.range(height, dtype=tf.float32) / tf.cast(height, tf.float32) + 1.0e-6
        xx = tf.range(width, dtype=tf.float32) / tf.cast(width, tf.float32) + 1.0e-6
        mesh_xx, mesh_yy = tf.meshgrid(xx*2.0*math.pi, yy*2.0*math.pi)
        mesh = tf.stack([mesh_yy, mesh_xx], axis=-1)
        mesh = tf.repeat(mesh[None, ...], batch_size, axis=0)

        dim_t = self._dim_t

        mesh = mesh[..., None] / dim_t
        mesh = tf.concat((tf.math.sin(mesh[..., 0::2]), tf.math.cos(mesh[..., 1::2])), axis=-1)
        mesh = tf.reshape(mesh, (batch_size, height, width, self.hidden_dim * 2))

        pos = self._token_projection(mesh, training=training)

        return pos

class ConvPatchEmbed(layers.Layer):
    ''' Image to patch embedding
    '''

    def __init__(self, patch_size=8, embed_dim=384, **kwargs):
        super().__init__(**kwargs)

        if patch_size != 16 and patch_size != 8 and patch_size != 4:
            raise('patch_size must be 4|8|16')

        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        n_ch = self.embed_dim
        activation = 'gelu'

        blocks = []
        if self.patch_size >= 16:
            blocks.append([
                layers.Conv2D(n_ch//8, 3, strides=2, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.Activation(activation),
            ])

        if self.patch_size >= 8:
            blocks.append([
                layers.Conv2D(n_ch//4, 3, strides=2, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.Activation(activation),
            ])
        blocks.append([
            layers.Conv2D(n_ch//2, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activation),
        ])
        blocks.append([
            layers.Conv2D(n_ch, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
        ])

        self._blocks = blocks

        super().build(input_shape)

    def call(self, x, training=None):
        y = x
        outputs = []
        for block in self._blocks:
            for op in block:
                y = op(y, training=training)
            outputs.append(y)

        return outputs

class LPI(layers.Layer):
    ''' Local patch interaction layer
    Use depth-wide conv to mixing feature from adjacient patches
    '''

    def __init__(self, kernel_size = 3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self._layers = [
            layers.DepthwiseConv2D(self.kernel_size, padding='same', activation='gelu'),
            layers.BatchNormalization(),
            layers.DepthwiseConv2D(self.kernel_size, padding='same'),
        ]

    def call(self, inputs, training=None):
        x, x_shape = inputs
        Hp, Wp = x_shape
        n_ch = tf.shape(x)[-1]
        y = tf.reshape(x, (-1, Hp, Wp, n_ch))
        for layer in self._layers:
            y = layer(y, training=training)
        y = tf.reshape(y, (-1, Hp * Wp, n_ch))

        return y

class MLP(layers.Layer):
    def __init__(self, filters, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.use_bias = use_bias

    def build(self, input_shape):
        in_channels = input_shape[-1]
        block = []
        for n_filter in self.filters:
            block.append(layers.Dense(n_filter, use_bias=self.use_bias, activation='gelu'))
        block.append(layers.Dense(in_channels, use_bias=self.use_bias))
        self._layers = block

        super().build(input_shape)

    def call(self, x, training=None):
        y = x
        for layer in self._layers:
            y = layer(y, training=training)
        return y

class XCA(layers.Layer):
    ''' Cross-Covariance Attention
    '''
    def __init__(self, n_heads=8, use_bias=False, **kwargs):
        super().__init__(**kwargs)

        self.n_heads = n_heads
        self.use_bias = use_bias

        self.temp = tf.Variable(tf.ones([n_heads, 1, 1], name='temperature'))

    def build(self, input_shape):
        n_ch = input_shape[-1]
        self._qkv_layer = layers.Dense(n_ch * 3, use_bias=self.use_bias)
        self._proj_layer = layers.Dense(n_ch)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        n_patches = tf.shape(x)[1]
        in_ch = x.shape[2]

        qkv = tf.reshape(self._qkv_layer(x, training=training), (batch_size, n_patches, self.n_heads, in_ch // self.n_heads, 3))
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = tf.linalg.normalize(q, axis=1)[0] # normalize along patch dim
        k = tf.linalg.normalize(k, axis=1)[0]

        q = tf.transpose(q, (0,2,3,1)) # (batch, heads, ch, patch)
        k = tf.transpose(k, (0,2,1,3)) # (batch, heads, patch, ch)
        v = tf.transpose(v, (0,2,3,1)) # (batch, heads, ch, patch)

        attn = tf.matmul(q, k) * self.temp
        attn = tf.math.softmax(attn, axis=-1)
        # attn = self._att_drop(attn)

        y = tf.transpose(tf.matmul(attn, v), (0,3,1,2)) #(batch, patch, head, ch)
        y = tf.reshape(y, (batch_size, -1, in_ch))

        y = self._proj_layer(y, training=training)
        # y = self._proj_drop(y)

        return y

class XCABlock(layers.Layer):
    def __init__(self, n_heads=8, use_bias=False, mlp_filters=1536, **kwargs):
        super().__init__(**kwargs)

        self._input_norm = layers.LayerNormalization()
        self._attn_norm = layers.LayerNormalization()
        self._lpi_norm = layers.LayerNormalization()
        self._attn_layer = XCA(n_heads=n_heads, use_bias=use_bias)
        self._lpi_layer = LPI()
        self._mlp_layer = MLP([mlp_filters,])

    def build(self, input_shape):
        dim = input_shape[0][-1]
        self.gamma1 = tf.Variable(tf.ones(dim), name='gamma1')
        self.gamma2 = tf.Variable(tf.ones(dim), name='gamma2')
        self.gamma3 = tf.Variable(tf.ones(dim), name='gamma3')

    def call(self, inputs, training=None):
        y, y_shape = inputs
        y = self._input_norm(y, training=training)
        y = self._attn_layer(y, training=training) * self.gamma1
        y = self._attn_norm(y, training=training)
        y = self._lpi_layer((y, y_shape), training=training) * self.gamma2
        y = self._lpi_norm(y, training=training)
        y = self._mlp_layer(y, training=training) * self.gamma3
        return y

class Decoder(layers.Layer):
    def __init__(self, method='conv_add', conv_repeats=2, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.conv_repeats = conv_repeats

    def build(self, input_shape):
        self._decoder_conv_list = []
        self._mixer_list = []
        for s in input_shape[:-1]:
            self._decoder_conv_list.append(self.build_conv_block(s[-1]))
            self._mixer_list.append(MixingBlock(method=self.method))

        super().build(input_shape)

    def build_conv_block(self, n_ch):
        conv_kwargs = {
            'padding': 'same',
            'kernel_initializer': 'he_normal',
            'activation': 'gelu'
        }
        block = []
        for k in range(self.conv_repeats):
            block.append(layers.Conv2D(n_ch, 3, **conv_kwargs))

        return block

    def call(self, inputs, training=None):
        outputs = [None] * len(inputs)
        outputs[-1] = inputs[-1]
        for k in range(len(inputs)-1, 0, -1):
            mixer = self._mixer_list[k-1]
            conv_block = self._decoder_conv_list[k-1]
            x = mixer((inputs[k-1], inputs[k]), training=training)
            for layer in conv_block:
                x = layer(x, training=training)
            outputs[k-1] = x

        return outputs

class XCiT(layers.Layer):
    def __init__(self, patch_size=8, embed_dim=384, depth=12, n_heads=8, use_bias=True, mlp_ratio=4, **kwargs):
        '''
        Args:
            patch_size:
            embed_dim:
            depth:
            n_heads:
            use_bias:
            mlp_ratio:
        '''

        super().__init__(**kwargs)
        self._config_dict = {
            'patch_size': patch_size,
            'embed_dim':embed_dim,
            'depth': depth,
            'n_heads': n_heads,
            'use_bias': use_bias,
            'mlp_ratio': mlp_ratio,
        }

    def get_config(self):
        return self._config_dict

    def build(self, input_shape):
        self._embed_layer = ConvPatchEmbed(
            patch_size=self._config_dict['patch_size'],
            embed_dim=self._config_dict['embed_dim'],
            )

        self._pos_embed_layer = PositionalEncodingFourier()

        mlp_filters = int(self._config_dict['embed_dim'] * self._config_dict['mlp_ratio'])
        blocks = []
        for i in range(self._config_dict['depth']):
            blocks.append(
                XCABlock(
                    n_heads=self._config_dict['n_heads'],
                    use_bias=self._config_dict['use_bias'],
                    mlp_filters=mlp_filters,
                )
            )
        self._att_blocks = blocks

        self._decoder = Decoder(method='upsampling')

        super().build(input_shape)

    def call(self, x, training=None):
        dim = self._config_dict['embed_dim']
        embed_out = self._embed_layer(x, training=training)
        y = embed_out[-1]
        pos_encoding = self._pos_embed_layer(y, training=training)
        y += pos_encoding

        Hp, Wp = tf.shape(y)[1], tf.shape(y)[2]
        y = tf.reshape(y, (-1, Hp*Wp, dim))
        for blk in self._att_blocks:
            y = blk((y, (Hp, Wp)), training=training)

        y = tf.reshape(y, (-1, Hp, Wp, dim))
        embed_out[-1] = y

        decoder_out = self._decoder(embed_out, training=training)
        keys = [str(k) for k in range(1, len(decoder_out)+1)]
        decoder_out = dict(zip(keys, decoder_out))

        return decoder_out
