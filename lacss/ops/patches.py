from typing import Tuple, Sequence, Union, Dict
import jax
from .boxes import box_iou_similarity
jnp = jax.numpy

def gather_patches(
    features, 
    locations, 
    patch_size: int
  ) -> tuple:
    ''' extract feature patches according to a list of locations
    Args:
        features: [H,W,C] standard 2D feature map
        locations: [N, 2] float32, scaled 0..1
        patch_size: int
    Returns:
        patches: [N, patch_size, patch_size, C] 
        yy: [N, patch_size, patch_size]: y coordinates of patches
        xx: [N, patch_size, patch_size]: x coordinates of patches
    '''
    height, width, _ = features.shape
    
    locations *= jnp.array([height, width])
    i_locations = (locations + .5).astype(int)
    i_locations_x = jnp.clip(i_locations[:, 1], 0, width-1)
    i_locations_y = jnp.clip(i_locations[:, 0], 0, height-1)
    yy, xx = jnp.mgrid[:patch_size, :patch_size] - patch_size // 2
    xx += i_locations_x[:, None, None]
    yy += i_locations_y[:, None, None]
    remainder = locations - jnp.stack([i_locations_y, i_locations_x], axis=-1)
    
    # padding to avoid out-of-bound
    padding_size = patch_size // 2 + 1
    paddings = [
        [padding_size, padding_size],
        [padding_size, padding_size],
        [0, 0],
    ]
    padded_features = jnp.pad(features, paddings)

    patches = padded_features[yy + padding_size, xx + padding_size, :]

    return patches, yy, xx, remainder

def _get_patch_data(predictions):
    if isinstance(predictions, dict):
        patches = predictions['instance_output']
        yc = predictions['instance_yc']
        xc = predictions['instance_xc']
    else:
        patches, yc, xc = predictions

    if patches.ndim > yc.ndim:
        patches = patches.squeeze(-1)

    return patches, yc, xc

def bboxes_of_patches(
    predictions: Union[Sequence, Dict],
    threshold: float = 0.5
) -> jnp.ndarray:
    '''
    Args:
        predictions: either a tuple or a dict containing three arrays:
            instance_outputs: [n, patch_size, patch_size, 1] float
            instance_y_coords: [n, patch_size, patch_size] y patch coords
            instance_x_coords: [n, patch_size, patch_size] x patch coords
        threshold: float
    Returns:
        bboxes: [n, 4] int
        bboox for empty patches are filled with -1
    '''
    patches, yy, xx = _get_patch_data(predictions)

    _, d0, d1 = patches.shape
    row_mask = jnp.any(patches>threshold, axis=1)
    col_mask = jnp.any(patches>threshold, axis=2)

    min_col = row_mask.argmax(axis=1)
    max_col = d1 - row_mask[:, ::-1].argmax(axis=1)

    min_row = col_mask.argmax(axis=1)
    max_row = d0 - col_mask[:, ::-1].argmax(axis=1)

    min_row += yy[:, 0, 0]
    max_row += yy[:, 0, 0]
    min_col += xx[:, 0, 0]
    max_col += xx[:, 0, 0]
    
    is_valid = row_mask.any(axis=1, keepdims=True)
    bboxes = jnp.stack([min_row, min_col, max_row, max_col], axis=-1)
    bboxes = jnp.where(is_valid, bboxes, -1)

    return bboxes

def indices_of_patches(
    predictions: Union[Sequence, Dict],
    image_shape: Tuple[int, int] = None, 
    threshold: float = 0.5
    ) -> tuple:
    '''
    Args:
        predictions: either a tuple or a dict containing three arrays:
            instance_outputs: [n, patch_size, patch_size, 1] float
            instance_y_coords: [n, patch_size, patch_size] y patch coords
            instance_x_coords: [n, patch_size, patch_size] x patch coords
        image_shape: (height, width) int
        threshold: float
    Returns:
        data for MaskIndices
    '''
    patches, yy, xx = _get_patch_data(predictions)

    indices = patches >= threshold
    yy = yy[indices]
    xx = xx[indices]
    rowids = jnp.argwhere(indices)[:,0]

    if image_shape is not None:
        height, width = image_shape
        valid_coords = ((yy >= 0) & (yy < height)) & ((xx >= 0) & (xx < width))
        yy = yy[valid_coords]
        xx = xx[valid_coords]
        rowids = rowids[valid_coords]

    return jnp.stack([yy, xx, rowids], axis=-1)

    # remove empty ones
    # rowlengths = mask_coords.row_lengths()
    # rowlengths = tf.boolean_mask(rowlengths, rowlengths>0)
    # mask_coords = tf.RaggedTensor.from_row_lengths(mask_coords.values, rowlengths)

def iou_patches_and_labels(predictions, labels, BLOCK_SIZE=128):
    ''' Compute iou between two sets of segmentations.
    The first set is defined by patches; the second by image labels
    Args:
        predictions: either a tuple or a dict containing three arrays:
            instance_mask: [n, patch_size, patch_size, 1] float
            instance_yc: [n, patch_size, patch_size] y patch coords
            instance_xc: [n, patch_size, patch_size] x patch coords
        labels: (height, width) int, bg_label == 0
    Returns:
        [n, m] iou values. 
    '''
    patches, yc, xc = _get_patch_data(predictions)
    patches = patches >= .5
    pred_areas = jnp.count_nonzero(patches, axis=(-1,-2))

    # this rely on JAX's out-of-bound indexing behavior
    padded_labels = jnp.pad(labels, [[1,1],[1,1]], constant_values=-1)
    gt_patches = padded_labels[yc+1, xc+1]
    max_indices = ((labels.max() - 1) // BLOCK_SIZE + 1) * BLOCK_SIZE
    all_gt_areas = jnp.count_nonzero(labels[:,:,None] == jnp.arange(max_indices) + 1, axis=(0, 1)) 

    ious = []
    for k in range(1, labels.max() + 1, BLOCK_SIZE):
        gt_p = gt_patches == jnp.arange(k, k+BLOCK_SIZE).reshape(-1,1,1,1) # [B, N, s, s]
        gt_areas = all_gt_areas[k-1:k+BLOCK_SIZE-1] 
        intersect = jnp.count_nonzero(gt_p & patches, axis=(-1,-2))  # [B, N]
        ious.append(intersect / (pred_areas + gt_areas[:, None] - intersect + 1.0e-8)) 

    ious = jnp.concatenate(ious, axis=0).transpose() # [N, B * b]
    ious = ious[:, :labels.max()]
    return ious

def patches_to_segmentations(predictions, image_size):
    ''' expand patches to the full image size
    Args:
        predictions: either a tuple or a dict containing three arrays:
        image_size: a tuple of (height, width) 
    Returns:
        segmentations: [n, height, width]
    '''
    patches, yc, xc = _get_patch_data(predictions)

    n_patches, patch_size, _  = yc.shape
    page_nums = jnp.arange(n_patches)
    segms = jnp.zeros((n_patches,) + image_size)
    segms = segms.at[page_nums[:, None, None], yc, xc].set(patches)

    return (segms >= 0.5).astype(int)
