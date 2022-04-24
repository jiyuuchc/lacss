import tensorflow as tf

def self_supervised_segmentation_losses(y_pred, mask, coords, lam=1.0, beta=-2.0):
    '''
    Args:
        y: [n_patches, patch_size, patch_size, 1]: float32, prediction 0..1
        mask: [img_height, img_width]: int32
        coords: [n_patches, patch_size, patch_size, 2]: int32 meshgrid coordinates
    '''
    if len(y_pred.shape) == 4:
        y_pred = tf.squeeze(y_pred, -1)
    patch_size = tf.shape(y_pred)[1]
    pad_size = patch_size//2 + 1
    coords = coords + pad_size
    paddings = [
        [pad_size, pad_size],
        [pad_size, pad_size],
    ]
    mask = tf.pad(mask, paddings)

    log_yi = tf.math.log(tf.clip_by_value(1.0 - y_pred, tf.keras.backend.epsilon(), 1.0))
    log_yi_sum = tf.scatter_nd(coords, log_yi, tf.shape(mask))
    log_yi_sum += (1.0 - tf.cast(mask, tf.float32)) * beta

    log_yi -= tf.gather_nd(log_yi_sum, coords)

    loss = - tf.reduce_mean(y_pred) + tf.reduce_mean(y_pred * log_yi) * lam

    return loss

# def cutter_loss_low_mem(y, coords, area_shape, mask = None, lam = 1.0):
#     '''
#     loss function that uses sparse tensor to save memory
#     '''
#     #area_shape = np.broadcast_to(area_shape, (coords.shape[-1],)) # in case area_shape is a number
#     patch_shape = tuple(y.shape[1:])
#
#     ind0 = tf.constant(list(np.ndindex(patch_shape)), dtype = tf.dtypes.int64)
#
#     y_pred = tf.sigmoid(y)
#     log_yi = tf.math.log(tf.clip_by_value(1.0 - y_pred, tf.keras.backend.epsilon(), 1.0))
#
#     def pad_img(cropped, offsets):
#       ind = ind0 + tf.constant(offsets, dtype = tf.dtypes.int64)
#       sp = tf.sparse.SparseTensor(ind, tf.reshape(cropped, [-1]), tf.constant(area_shape + patch_shape, dtype = tf.dtypes.int64))
#       return tf.sparse.expand_dims(sp, 0)
#
#     log_yi_padded = tf.sparse.concat(
#       0,
#       [pad_img(cropped, coord) for cropped,coord in zip(tf.unstack(log_yi), list(coords))]
#     )
#
#     log_yi_sum = tf.sparse.reduce_sum(log_yi_padded, axis = 0)
#     if mask is not None:
#       log_yi_sum += mask
#
#     ind = ind0 + tf.constant(coords, dtype = tf.dtypes.int64)[:, tf.newaxis, :]
#     ind = tf.reshape(ind, tuple(log_yi.shape) + (len(patch_shape),))
#     log_yi -= tf.gather_nd(log_yi_sum, ind)
#
#     loss = - tf.reduce_sum(y_pred)
#     loss += lam * tf.reduce_sum(y_pred * log_yi) * lam
#
#     return loss /  y.shape[0]
