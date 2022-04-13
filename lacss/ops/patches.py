import tensorflow as tf

def gather_patches(features, locations, patch_size):
    ''' extract feature patches according to a list of locations
    Args:
        features: [batch_size, H,W,C] or [H,W,C] standard 2D feature map
        locations: [batch_size, N, 2] or [N, 2] int32 tensor, center coordiantes of each patch
        patch_size: int32
    Returns:
        patches: [batch_size, N, patch_size, patch_size] or [N, patch_size, patch_size]
    '''

    batched_op = True
    if len(features.shape) == 3:
        batched_op = False
        feaures = tf.expand_dims(features, 0)
        locations = tf.exapnd_dims(locations, 0)

    if len(features.shape) != 4 or len(locations.shape) != 4:
        raise ValueError

    rr = tf.range(patch_size) - patch_size // 2
    mesh = tf.stack(tf.meshgrid(rr, rr) , axis=-1)
    mesh_co = locations[:, :, None, None, :] + mesh

    # padding to avoid out-of-bound
    padding_size = patch_size // 2 + 1
    mesh_co = mech_co + padding_size
    paddings = [
        [0, 0],
        [padding_size, padding_size],
        [padding_size, padding_size],
        [0, 0],
    ]
    padded_features = tf.pad(features, paddings)

    patches = tf.gather_nd(features, mesh_co)

    if not batched_op:
        patches = tf.squeeze(patches, 0)

    return patches

def ragged_gather_patches(features, locations, patch_size):
    ''' batch extracting feature patches according to a ragged list of locations
    Args:
        features: [batch_size, H,W,C] standard 2D feature map
        locations: [batch_size, None, 2] int32 ragged tensor, center coordiantes of each patch
        patch_size: int32
    Returns:
        patches: [batch_size, None, patch_size, patch_size] ragged tensor
    '''

    patches = tf.map_fn(
        lambda x: gather_patches(x[0], x[1], patch_size),
        (features, locations),
        fn_output_signature = tf.RaggedTensorSpec([None, patch_size, patch_size, features.shape[-1]], features.dtype, 0),
    )

    return patches
