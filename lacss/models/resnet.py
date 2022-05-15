import tensorflow as tf
import tensorflow.keras.layers as layers

def build_resnet_backbone(input_shape=(None,None,1), is_v2=True):
    if is_v2:
        resnet = tf.keras.applications.ResNet50V2(include_top=False)
        rn_input = resnet.get_layer('pool1_pad').input
        rn_output = (
            resnet.get_layer('conv2_block3_preact_relu').output,
            resnet.get_layer('conv3_block4_preact_relu').output,
            resnet.get_layer('conv4_block6_preact_relu').output,
            resnet.get_layer('post_relu').output,
        )
    else:
        resnet = tf.keras.applications.ResNet50(include_top=False)
        rn_input = resnet.get_layer('pool1_pad').input
        rn_output = (
            resnet.get_layer('conv2_block3_out').output,
            resnet.get_layer('conv3_block4_out').output,
            resnet.get_layer('conv4_block6_out').output,
            resnet.get_layer('conv5_block3_out').output,
        )
    encoder = tf.keras.Model(inputs=rn_input, outputs=rn_output)
    input = tf.keras.layers.Input(shape=input_shape)
    x = input
    x = tf.keras.layers.Conv2D(24,3,padding='same', activation='relu',name='stem_conv1')(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same', activation='relu',name='stem_conv2')(x)
    x = encoder(x)

    x = [tf.keras.layers.Conv2D(256,1,activation='relu',name=f'fpn_conv{k}_1')(d) for k,d in enumerate(x)]

    m4 = tf.keras.layers.UpSampling2D(name='upsample_4')(x[-1]) + x[-2]
    out4 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', name=f'fpn_conv4_2')(m4)

    m3 = tf.keras.layers.UpSampling2D(name='upsample_3')(out4) + x[-3]
    out3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', name=f'fpn_conv3_2')(m3)

    m2 = tf.keras.layers.UpSampling2D(name='upsample_2')(out3) + x[-4]
    out2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', name=f'fpn_conv2_2')(m2)

    return tf.keras.Model(inputs=input, outputs=(out2, out4))
