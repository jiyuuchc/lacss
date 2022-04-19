import tensorflow as tf
from ..ops import *
from .iis import *

def generate_indicators(gt_locations, target_shape, scale_factor, high_threshold, low_threshold):
    '''
    Args:
      gt_locations: [batch_size, None, 2]
      target_shape: (height, width)
      scale_factor: int32
      high_threshold: min distance from cell center to be considered background
      low_threshold: max distance from cell center to be considered forground
    Returns:
      indicators: [batch_size, height, width], 0: background, 1: foreground, -1:ignore
    '''
    height, width = target_shape
    # high_threshold = 64.0
    # low_threshold = 12.0

    flat_locations = gt_locations.to_tensor(-1.0) / scale_factor
    batch_size = tf.shape(flat_locations)[0]
    xc, yx = tf.meshgrid(tf.range(width, dtype=tf.float32)+0.5, tf.range(height, dtype=tf.float32)+0.5)
    mesh = tf.stack([yx,xc], axis=-1)
    mesh = tf.repeat(mesh[None,...], batch_size, axis=0)
    mesh = tf.reshape(mesh, [batch_size, -1, 2])

    _, indicators = location_matching(mesh, flat_locations, [high_threshold, low_threshold], [0, -1, 1])
    indicators = tf.reshape(indicators, [batch_size, height, width])

    return indicators

def supervised_loss(indicators, pred):
    batch_size = tf.shape(pred)[0]

    if len(pred.shape) == 3:
        pred = tf.expand_dims(pred, -1)

    loss = tf.keras.losses.binary_crossentropy(indicators[..., None], pred)

    mask = tf.reshape(indicators >= 0, [batch_size, -1])
    loss = tf.reshape(loss, [batch_size, -1])
    loss = tf.reduce_sum(tf.where(mask, loss, 0), axis=-1) / tf.cast(tf.math.count_nonzero(mask, axis=-1), tf.float32)

    return tf.reduce_sum(loss)

def unsupervised_loss(pred_a, pred_b):
    _, _, _, n_ch = pred_a.shape.as_list()
    if n_ch == 1:
        pred_a = tf.concat([pred_a, 1.0 - pred_a], axis=-1)
        pred_b = tf.concat([pred_b, 1.0 - pred_b], axis=-1)
    return iis_loss(pred_a, pred_b)
