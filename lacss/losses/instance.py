import tensorflow as tf

def self_supervised_segmentation_losses(y_pred, coords, binary_mask, lam=1.0, beta=-2.):
    '''
    Args:
        y: [n_patches, patch_size, patch_size, 1]: float32, prediction 0..1
        coords: [n_patches, patch_size, patch_size, 2]: int32 meshgrid coordinates
        binary_mask: [img_height, img_width]: int32
    '''
    if len(y_pred.shape) == 4:
        y_pred = tf.squeeze(y_pred, -1)
    if len(binary_mask.shape) == 3:
        binary_mask = tf.squeeze(binary_mask, -1)

    patch_shape = tf.shape(y_pred, out_type=tf.int64)
    patch_size = tf.shape(y_pred)[1]

    padding_size = patch_size//2 + 2
    coords = coords + padding_size
    paddings = [
        [padding_size, padding_size],
        [padding_size, padding_size],
    ]
    mask = tf.pad(tf.cast(binary_mask, tf.float32), paddings)

    log_yi = tf.math.log(tf.clip_by_value(1.0 - y_pred, tf.keras.backend.epsilon(), 1.0))
    log_yi_sum = tf.scatter_nd(coords, log_yi, tf.shape(mask))
    log_yi_sum += (1.0 - mask) * beta

    log_yi = tf.gather_nd(log_yi_sum, coords) - log_yi

    loss = (tf.reduce_sum(binary_mask)-tf.reduce_sum(y_pred))/tf.cast(tf.size(y_pred), tf.float32) - tf.reduce_mean(y_pred * log_yi) * lam
    # loss = loss * tf.cast(patch_shape[0], loss.dtype)

    return loss

def self_supervised_segmentation_losses_2(y_pred, coords, binary_mask, lam=1.0, beta=-2.):
    '''
    Args:
        y: [n_patches, patch_size, patch_size, 1]: float32, prediction 0..1
        coords: [n_patches, patch_size, patch_size, 2]: int32 meshgrid coordinates
        binary_mask: [img_height, img_width]: int32
    '''
    if len(y_pred.shape) == 4:
        y_pred = tf.squeeze(y_pred, -1)
    if len(binary_mask.shape) == 3:
        binary_mask = tf.squeeze(binary_mask, -1)

    patch_shape = tf.shape(y_pred, out_type=tf.int64)
    patch_size = tf.shape(y_pred)[1]

    padding_size = patch_size//2 + 2
    coords = coords + padding_size
    paddings = [
        [padding_size, padding_size],
        [padding_size, padding_size],
    ]
    mask = tf.pad(tf.cast(binary_mask, tf.float32), paddings)

    log_yi = tf.math.log(tf.clip_by_value(1.0 - y_pred, tf.keras.backend.epsilon(), 1.0))
    log_yi_sum = tf.scatter_nd(coords, log_yi, tf.shape(mask))

    binary_loss = -(1.0 - mask) * log_yi_sum - mask * tf.math.log(1.0 + tf.keras.backend.epsilon() - tf.math.exp(log_yi_sum))
    binary_loss = tf.reduce_mean(binary_loss)

    log_yi = tf.gather_nd(log_yi_sum, coords) - log_yi

    overlap_loss =  - tf.reduce_mean(y_pred * log_yi)

    loss = (binary_loss + lam * overlap_loss) * tf.cast(patch_shape[0], loss.dtype)

    return loss

def supervised_segmentation_losses(y_pred, coordinates, mask_indices):
    '''
    Args:
        y: [n_patches, patch_size, patch_size, 1]: float32, prediction 0..1
        coordinates: [n_patches, patch_size, patch_size, 2] int32 meshgrid coordinates
        mask_indices: [n_patches, None, 2] int64 Ragged tensor
    return:
        loss: based on cross entropy
    '''
    patch_shape = tf.shape(y_pred, out_type=tf.int64)
    roi_size = patch_shape[1]
    #top left
    c0 = coordinates[:,0,0,:]
    c0 = tf.repeat(c0, mask_indices.row_lengths(), axis=0)

    mi_flatten = mask_indices.values - tf.cast(c0, tf.int64)
    mi_flatten = tf.concat([mask_indices.value_rowids()[:, None], mi_flatten], axis=-1)
    mask = tf.logical_and(
        tf.logical_and(mi_flatten[:,1] >= 0, mi_flatten[:,1] < roi_size),
        tf.logical_and(mi_flatten[:,2] >= 0, mi_flatten[:,2] < roi_size),
        )
    mi_flatten = tf.boolean_mask(mi_flatten, mask)

    n_pixels = tf.shape(mi_flatten)[0]
    gt_patches = tf.scatter_nd(mi_flatten, tf.ones([n_pixels], tf.int32), shape=patch_shape[:3])
    gt_patches = tf.expand_dims(gt_patches, -1)

    loss = tf.keras.losses.binary_crossentropy(gt_patches, y_pred)
    loss = tf.reduce_mean(loss) * tf.cast(patch_shape[0], loss.dtype)

    return loss

def self_supervised_edge_losses(patch_pred, coords, edge_pred):

    patch_shape = tf.shape(patch_pred, out_type=tf.int64)

    # patch_edges = tf.math.sqrt(tf.reduce_mean(tf.image.sobel_edges(patch_pred) ** 2 / 4, axis=-1))
    patch_edges = tf.math.square(tf.image.sobel_edges(patch_pred))
    patch_edges = (patch_edges[..., 0] + patch_edges[..., 1]) / 8.0

    # quite strange this is needed.. GPU error
    patch_edges = tf.clip_by_value(patch_edges, 1e-9, 1.0)
    patch_edges = tf.math.sqrt(patch_edges)
    combined_edges = tf.scatter_nd(coords, patch_edges[...,0], tf.shape(edge_pred))
    combined_edges = tf.math.tanh(combined_edges)

    loss = tf.keras.losses.huber(edge_pred, combined_edges, delta=.5)

    # return tf.reduce_mean(loss) * tf.cast(patch_shape[0], loss.dtype)
    return tf.reduce_mean(loss)
