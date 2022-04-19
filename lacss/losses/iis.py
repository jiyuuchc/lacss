import sys
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers

def image_iis_loss(y0, y1, neighbours=1 ):
    ''' mutual information loss for images
        Here the loss is computed not only between corressponding pixels of y0 and y1, but also between
        neighbouring pixels.
    '''
    loss = 0.0
    batch_size, h, w, n_cls = tf.shape(y0)

    joint_prob = tf.zeros([n_cls, n_cls], y0.dtype)
    for dx in range(-neighbours, neighbours+1):
        for dy in range(-neighbours, neighbours+1):
            y0f = tf.image.crop_to_bounding_box(y0, 0, 0, h - dy, w - dx)
            y1f = tf.image.crop_to_bounding_box(y1, dy, dx, h - dy, w - dx)
            y0f = tf.reshape(y0f, [-1, 1, n_cls])
            y1f = tf.reshape(y1f, [-1, n_cls, 1])

            joint_prob += tf.reduce_mean(y0f * y1f, axis = 0)

    n_neighbours = (2 * neightbours + 1) * (2 * neightbours + 1)
    joint_prob = joint_prob / n_neighbours
    joint_prob = tf.clip_by_value(joint_prob, sys.float_info.epsilon, 1e9)

    log_marginal = tf.math.log(tf.reduce_sum(joint_prob, axis = 0))

    loss = - tf.reduce_sum(joint_prob * (tf.math.log(joint_prob) - log_marginal[None, :] - log_marginal[:, None]))

    return loss

def iis_loss(y0, y1):
    ''' mutual information loss
        y0, y0:  last dimension is the number of classes, ie. shape = [..., n_cls]
    '''
    n_cls = y0.shape[-1]
    if n_cls != y1.shape[-1]:
        raise ValueError

    y0 = tf.reshape(y0, [-1, 1, n_cls])
    y1 = tf.reshape(y1, [-1, n_cls, 1])

    # averge over batch instead of sum of batch
    joint_prob = tf.reduce_mean(y0 * y1, axis = 0)
    joint_prob += tf.transpose(joint_prob) #make symetric
    joint_prob = tf.clip_by_value(joint_prob, sys.float_info.epsilon, 1e9)

    # only need to compute log_marginal for one axis, since joint_prob is symmetric
    log_marginal = tf.math.log(tf.reduce_sum(joint_prob, axis = 0))

    loss = -tf.reduce_sum(joint_prob * (tf.math.log(joint_prob) - log_marginal[None, :] - log_marginal[:, None]))

    return loss

class IISLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(IISLoss, tf.keras.losses.Loss).__init__(**kwargs)

    def call(self, y_true, y_pred):
        y0, y1 = y_pred
        loss = iis_loss(y0, y1)
        return loss

class IISImageLoss(tf.keras.losses.Loss):
    def __init__(self, neighbours = 1, **kwargs):
        super(IISImageLoss, tf.keras.losses.Loss).__init__(kwargs)
        self._config_dict = kwargs
        self._config_dict.update({'neighbours': neighbours})

    def get_config(self):
        return self._config_dict

    def call(self, y_true, y_pred):
        y0, y1 = y_pred
        loss = iis_image_loss(y0, y1, self._config_dict['neighbours'])

        return loss
