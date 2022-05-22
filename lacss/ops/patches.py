import tensorflow as tf

def make_meshgrids(i_locations:int, patch_size:int):
    rr = tf.range(patch_size, dtype=tf.int32) - patch_size // 2
    xx,yy = tf.meshgrid(rr, rr)
    mesh = tf.stack([yy,xx], axis=-1)
    mesh_co = i_locations[..., None, None, :] + mesh
    return mesh_co

def gather_patches(features, locations, patch_size):
    ''' extract feature patches according to a list of locations
    Args:
        features: [batch_size, H,W,C] or [H,W,C] standard 2D feature map
        locations: [batch_size, N, 2] or [N, 2] float32 tensor, 0..1
        patch_size: int32
    Returns:
        patches: [batch_size, N, patch_size, patch_size] or [N, patch_size, patch_size]
        mesh: [batch_size, N, patch_size, patch_size, 2] or [N, patch_size, patch_size, 2]: coordinates of patches
    '''

    batched_op = True
    if len(features.shape) == 3:
        batched_op = False
        features = tf.expand_dims(features, 0)
        locations = tf.expand_dims(locations, 0)

    if len(features.shape) != 4 or len(locations.shape) != 3:
        raise ValueError

    # locations = tf.clip_by_value(locations, 0, 1)
    locations = locations * tf.cast(tf.shape(features)[1:3], locations.dtype)
    i_locations = tf.cast(locations, tf.int32)
    mesh_co = make_meshgrids(i_locations, patch_size)

    # padding to avoid out-of-bound
    padding_size = patch_size // 2 + 1
    paddings = [
        [0, 0],
        [padding_size, padding_size],
        [padding_size, padding_size],
        [0, 0],
    ]
    padded_features = tf.pad(features, paddings)

    patches = tf.gather_nd(padded_features, mesh_co + padding_size, batch_dims=1)

    if not batched_op:
        patches = tf.squeeze(patches, 0)
        mesh_co = tf.squeeze(mesh_co, 0)

    return patches, mesh_co

# def ragged_gather_patches(features, locations, patch_size):
#     ''' batch extracting feature patches according to a ragged list of locations
#     Args:
#         features: [batch_size, H,W,C] standard 2D feature map
#         locations: [batch_size, None, 2] float32 ragged tensor, 0..1
#         patch_size: int32
#     Returns:
#         patches: [batch_size, None, patch_size, patch_size] ragged tensor
#     '''
#
#     patches = tf.map_fn(
#         lambda x: gather_patches(x[0], x[1], patch_size)[0],
#         (features, locations),
#         fn_output_signature = tf.RaggedTensorSpec([None, patch_size, patch_size, features.shape[-1]], features.dtype, 0),
#     )
#
#     return patches

def bboxes_of_patches(patches, patch_coords, threshold=0.5):
    '''
    Args:
        patches: [n, patch_size, patch_size, 1] float
        patch_coords: [n, patch_size, patch_size, 1]
        threshold: float
    Returns:
        bboxes: int64
    '''
    patches = tf.squeeze(patches, -1)
    _, d0, d1 = patches.shape
    row_mask = tf.reduce_any(patches>threshold, axis=1)
    col_mask = tf.reduce_any(patches>threshold, axis=2)
    is_valid = tf.reduce_any(row_mask, axis=1, keepdims=True)

    min_col = tf.argmax(row_mask, axis=1)
    max_col = d1 - tf.argmax(tf.reverse(row_mask, [1]), axis=1)

    min_row = tf.argmax(col_mask, axis=1)
    max_row = d0 - tf.argmax(tf.reverse(col_mask, [1]), axis=1)

    min_row += tf.cast(patch_coords[:, 0, 0, 0], tf.int64)
    max_row += tf.cast(patch_coords[:, 0, 0, 0], tf.int64)
    min_col += tf.cast(patch_coords[:, 0, 0, 1], tf.int64)
    max_col += tf.cast(patch_coords[:, 0, 0, 1], tf.int64)

    bboxes = tf.stack([min_row, min_col, max_row, max_col], -1)
    bboxes = tf.where(is_valid, bboxes, -1)

    return bboxes

def indices_of_patches(patches, patch_coords, image_shape=None, threshold=0.5):
    instance_output = tf.squeeze(patches, -1)
    indices = tf.where(instance_output >= threshold)
    mask_coords = tf.gather_nd(patch_coords, indices)
    rowids = indices[:,0]

    if image_shape is not None:
        height, width = image_shape
        valid_coords = tf.logical_and(
            tf.logical_and(mask_coords[:,0]>=0, mask_coords[:,0] < height),
            tf.logical_and(mask_coords[:,1]>=0, mask_coords[:,1] < width),
        )
        mask_coords = tf.boolean_mask(mask_coords, valid_coords)
        rowids = tf.boolean_mask(rowids, valid_coords)

    mask_coords = tf.RaggedTensor.from_value_rowids(mask_coords, rowids)
    mask_coords = tf.cast(mask_coords, tf.int64)
    # remove empty ones
    # rowlengths = mask_coords.row_lengths()
    # rowlengths = tf.boolean_mask(rowlengths, rowlengths>0)
    # mask_coords = tf.RaggedTensor.from_row_lengths(mask_coords.values, rowlengths)

    return mask_coords
