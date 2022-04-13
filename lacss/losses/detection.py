import tensorflow as tf

def locations_to_labels(locations, target_shape, scale_factor, threshold):
    ''' Generate labels as score regression targets
    Args:
        locations: [batch_size, None, 2] float32 RaggedTensor
        target_shape: (h, w) label size
        scale_factor: down_scale of the label size relative to original input
        threshold: distance threshold for postive label
    Returns:
        score_target: [batch_size, h, w, 1] int32 tensor
        regression_target: [batch_size, h, w, 2] float tensor
    '''
    height, width = target_shape

    flat_locations = locations.to_tensor(-1.0) / scale_factor
    mask = flat_locations[...,0] > 0
    xc, yc = tf.meshgrid(tf.range(width, dtype=tf.float32)+0.5, tf.range(height, dtype=tf.float32)+0.5)
    mesh = tf.stack([yc, xc], axis=-1)
    mesh = tf.repeat(mesh[None,...], tf.shape(flat_locations)[0], axis = 0)
    distances = flat_locations[:, :, None, None, :] - mesh[:, None, :, :, :]
    distances_sq = tf.reduce_sum(distances * distances, axis=-1)
    distances_sq = tf.where(mask[..., None, None], distances_sq, float('inf'))
    indices = tf.cast(tf.argmin(distances_sq, axis=1), tf.int32)
    best_distances = tf.experimental.numpy.take_along_axis(distances_sq, indices[:,None,:,:], 1)
    best_distances = tf.squeeze(best_distances, 1)

    score_target = tf.where(best_distances < 1.5, 1, 0)
    score_target = tf.expand_dims(score_target, -1)

    indices = tf.repeat(indices[:, None, :, :, None], 2, axis=-1)
    regression_target = tf.experimental.numpy.take_along_axis(distances, indices, 1)
    regression_target = tf.squeeze(regression_target, 1)

    return score_target, regression_target

def detection_losses(gt, pred, scale_factor=8, threshold=1.5, **kwargs):
    pred_scores, pred_regressions = pred
    img_height = tf.shape(pred_scores)[-3]
    img_width = tf.shape(pred_scores)[-2]
    batch_size = tf.shape(pred_scores)[0]

    score_target, regression_target = locations_to_labels(gt, (img_height, img_width), scale_factor, threshold)

    score_loss = tf.keras.losses.binary_focal_crossentropy(score_target, pred_scores, **kwargs)
    score_loss = tf.reduce_sum(tf.reduce_mean(tf.reshape(score_loss, [batch_size, -1]), axis=1))

    regression_loss = tf.keras.losses.huber(regression_target, pred_regressions)
    regression_loss = tf.reshape(regression_loss, [batch_size, -1])
    score_target = tf.reshape(score_target, [batch_size, -1])
    regression_loss = tf.reduce_sum(tf.where(score_target>0, regression_loss, 0), axis=1) / (tf.cast(tf.reduce_sum(score_target, axis=1), tf.float32) + 1e-9)
    regression_loss = tf.reduce_sum(regression_loss)

    return score_loss, regression_loss
