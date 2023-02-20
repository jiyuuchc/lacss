from typing import Tuple, Sequence, Union, Dict
import jax
from .boxes import box_iou_similarity
jnp = jax.numpy

''' Various functions deals with segmentation pathces
    All functions here takes unbatched input. Use vmap to convert to batched data
'''

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
        y0: [N]: y0 coordinates of patches
        x0: [N]: x0 coordinates of patches
    '''
    height, width, _ = features.shape
    
    locations *= jnp.array([height, width])
    #i_locations = (locations + .5).astype(int)
    i_locations = locations.astype(int)
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

    return patches, yy[:,0,0], xx[:,0,0], remainder

def _get_patch_data(pred):
    if isinstance(pred, dict):
        patches = pred['instance_output']
        yc = pred['instance_yc']
        xc = pred['instance_xc']
    else:
        patches, yc, xc = pred

    if patches.ndim > yc.ndim:
        patches = patches.squeeze(-1)

    return patches, yc, xc

def bboxes_of_patches(
    pred: Union[Sequence, Dict],
    threshold: float = 0.5
) -> jnp.ndarray:
    '''
    Args:
        pred: either a tuple or a dict containing three arrays:
            instance_outputs: [n, patch_size, patch_size, 1] float
            instance_y_coords: [n, patch_size, patch_size] y patch coords
            instance_x_coords: [n, patch_size, patch_size] x patch coords
        threshold: float
    Returns:
        bboxes: [n, 4] int
        bboox for empty patches are filled with -1
    '''
    patches, yy, xx = _get_patch_data(pred)

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
    pred: Union[Sequence, Dict],
    input_size: Tuple[int, int] = None,
    threshold: float = 0.5
    ) -> tuple:
    '''
    Args:
        pred: model output (unbatched):
        threshold: float
    Returns:
        data for MaskIndices
    '''
    patches, yy, xx = _get_patch_data(pred)

    indices = patches >= threshold
    yy = yy[indices]
    xx = xx[indices]
    rowids = jnp.argwhere(indices)[:,0]

    if input_size is not None:
        height, width = input_size
        valid_coords = ((yy >= 0) & (yy < height)) & ((xx >= 0) & (xx < width))
        yy = yy[valid_coords]
        xx = xx[valid_coords]
        rowids = rowids[valid_coords]

    return jnp.stack([yy, xx, rowids], axis=-1)

def iou_patches_and_labels(pred, labels, BLOCK_SIZE=128):
    ''' Compute iou between two sets of segmentations.
    The first set is defined by patches; the second by image labels
    Args:
        pred: model output (unbatched):
        threshold: float
    Returns:
        [n, m] iou values. 
    '''
    patches, yc, xc = _get_patch_data(pred)
    patches = patches >= .5
    pred_areas = jnp.count_nonzero(patches, axis=(-1,-2))

    # this rely on JAX's out-of-bound indexing behavior
    padded_labels = jnp.pad(labels, [[1,1],[1,1]], constant_values=-1)
    gt_patches = padded_labels[yc+1, xc+1]
    max_indices = ((labels.max() - 1) // BLOCK_SIZE + 1) * BLOCK_SIZE
    all_gt_areas = jnp.count_nonzero(labels[:,:,None] == jnp.arange(max_indices) + 1, axis=(0, 1)) 

    ious = []
    # FIXME change to scan
    for k in range(1, labels.max() + 1, BLOCK_SIZE):
        gt_p = gt_patches == jnp.arange(k, k+BLOCK_SIZE).reshape(-1,1,1,1) # [B, N, s, s]
        gt_areas = all_gt_areas[k-1:k+BLOCK_SIZE-1] 
        intersect = jnp.count_nonzero(gt_p & patches, axis=(-1,-2))  # [B, N]
        ious.append(intersect / (pred_areas + gt_areas[:, None] - intersect + 1.0e-8)) 

    ious = jnp.concatenate(ious, axis=0).transpose() # [N, B * b]
    ious = ious[:, :labels.max()]
    return ious

def patches_to_segmentations(pred, input_size, threshold=0.5):
    ''' expand patches to the full image size
    Args:
        pred: model output (unbatched):
        input_size: tuple(int, int)
        threshold: float
    Returns:
        segmentations: [n, height, width]
    '''
    patches, yc, xc = _get_patch_data(pred)
    n_patches, patch_size, _  = yc.shape

    page_nums = jnp.arange(n_patches)
    segms = jnp.zeros((n_patches,) + input_size)
    segms = segms.at[page_nums[:, None, None], yc, xc].set(patches)

    return (segms >= threshold).astype(int)

def patches_to_label(pred, input_size, score_threshold=0.5, threshold=0.5):
    ''' convert patches to the image label
    Args:
        pred: model output (unbatched):
        threshold: float
    Returns:
        label: [height, width]
    '''
    label = jnp.zeros(input_size)

    if 'training_locations' in pred:
        mask = pred['instance_mask']
    else:
        mask = pred['pred_scores'] >= score_threshold

    pr = pred['instance_output']
    n = jnp.count_nonzero(pred['instance_mask'])
    seq = jnp.arange(pr.shape[0])
    seq = n - jnp.where(seq<n, seq, n)
    pr = (pr > threshold) * (seq[:, None, None])
    yc = pred['instance_yc']
    xc = pred['instance_xc']

    label = label.at[yc, xc].max(pr)

    return label
