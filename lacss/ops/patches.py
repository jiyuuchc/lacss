import tensorflow as tf

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
        feaures = tf.expand_dims(features, 0)
        locations = tf.exapnd_dims(locations, 0)

    if len(features.shape) != 4 or len(locations.shape) != 3:
        raise ValueError

    # locations = tf.clip_by_value(locations, 0, 1)
    locations = locations * tf.cast(tf.shape(features)[1:3], locations.dtype)
    i_locations = tf.cast(locations + 0.5, tf.int32)

    rr = tf.range(patch_size, dtype=tf.int32) - patch_size // 2
    mesh = tf.stack(tf.meshgrid(rr, rr) , axis=-1)
    mesh_co = i_locations[:, :, None, None, :] + mesh

    # padding to avoid out-of-bound
    padding_size = patch_size // 2 + 1
    paddings = [
        [0, 0],
        [padding_size, padding_size],
        [padding_size, padding_size],
        [0, 0],
    ]
    padded_features = tf.pad(features, paddings)

    patches = tf.gather_nd(features, mesh_co + padding_size, batch_dims=1)

    if not batched_op:
        patches = tf.squeeze(patches, 0)
        mech_co = tf.squeeze(mech_co, 0)

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
