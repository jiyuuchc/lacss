import tensorflow as tf

def locations_to_labels(locations, target_shape, threshold=1.5):
    ''' Generate labels as score regression targets
    Args:
        locations: [batch_size, N, 2] float32 true location values. scaled 0..1
        target_shape: [2,]  a int tensor indicating the expected output size
        threshold: distance threshold for postive label
    Returns:
        score_target: [batch_size, h, w, 1] int32 tensor
        regression_target: [batch_size, h, w, 2] float tensor
    '''
    height = target_shape[0]
    width = target_shape[1]
    scaling = tf.cast(target_shape, locations.dtype)

    if type(locations) is tf.RaggedTensor:
        flat_locations = locations.to_tensor(-1.0) * scaling
    else:
        flat_locations = locations * scaling

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

    score_target = tf.cast(tf.where(best_distances < threshold, 1, 0), tf.float32)
    score_target = tf.expand_dims(score_target, -1)

    indices = tf.repeat(indices[:, None, :, :, None], 2, axis=-1)
    regression_target = tf.experimental.numpy.take_along_axis(distances, indices, 1)
    regression_target = tf.squeeze(regression_target, 1)

    return score_target, regression_target

def detection_losses(gt_locations, pred_scores, pred_regressions, roi_size=1.5, **kwargs):
    '''
    Args:
        gt_locations: [batch_size, N, 2] float32 true location values. scaled 0..1
        pred_scores: [batch_size, h, w, 1] detector score output
        pred_regressions: [batch_size, h, w, 2] detector regression output
        roi_size: float number, the size of detection ROI
    Returns:
        score_loss: float
        regression_loss: float
    '''
    # pred_scores, pred_regressions = pred

    target_shape = tf.shape(pred_scores)[1:3]
    batch_size = tf.shape(pred_scores)[0]

    score_target, regression_target = locations_to_labels(gt_locations, target_shape, roi_size)

    score_loss = tf.keras.losses.binary_focal_crossentropy(score_target, pred_scores, **kwargs)
    score_loss = tf.reduce_mean(score_loss)
    # score_loss = tf.reduce_sum(tf.reduce_mean(tf.reshape(score_loss, [batch_size, -1]), axis=1))

    regression_loss = tf.keras.losses.huber(regression_target, pred_regressions)
    regression_loss = tf.reshape(regression_loss, [batch_size, -1])
    score_target = tf.reshape(score_target, [batch_size, -1])
    regression_loss = tf.reduce_sum(tf.where(score_target>0, regression_loss, 0), axis=1) / (tf.reduce_sum(score_target, axis=1)+ 1e-9)
    regression_loss = tf.reduce_mean(regression_loss)
    # regression_loss = tf.reduce_sum(regression_loss)

    return score_loss, regression_loss
