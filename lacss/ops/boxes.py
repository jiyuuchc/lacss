import jax
jnp = jax.numpy

def box_area(box):
    """Computes area of boxes.
    Args:
      box: a float Tensor with [N, 4], or [B, N, 4].

    Returns:
      a float Tensor with [N], or [B, N]
    """
    y_min, x_min, y_max, x_max = jnp.split(box, 4, axis=-1)
    return jnp.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)

def box_intersection(gt_boxes, boxes):
    """Compute pairwise intersection areas between boxes.

    Args:
      gt_boxes: [..., N, 4]
      boxes: [..., M, 4]

    Returns:
      a float Tensor with shape [..., N, M] representing pairwise intersections.
    """
    y_min1, x_min1, y_max1, x_max1 = jnp.split(gt_boxes, 4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = jnp.split(boxes, 4, axis=-1)

    # [N, M] or [B, N, M]
    y_min_max = jnp.minimum(y_max1, y_max2.swapaxes(-1, -2))
    y_max_min = jnp.maximum(y_min1, y_min2.swapaxes(-1, -2))
    x_min_max = jnp.minimum(x_max1, x_max2.swapaxes(-1, -2))
    x_max_min = jnp.maximum(x_min1, x_min2.swapaxes(-1, -2))

    intersect_heights = y_min_max - y_max_min
    intersect_widths = x_min_max - x_max_min
    intersect_heights = jnp.maximum(0, intersect_heights)
    intersect_widths = jnp.maximum(0, intersect_widths)

    return intersect_heights * intersect_widths

def box_iou_similarity(gt_boxes, boxes):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      gt_boxes: a float Tensor with [N, 4].
      boxes: a float Tensor with [M, 4].

    Returns:
      a Tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = box_intersection(gt_boxes, boxes)
    gt_boxes_areas = box_area(gt_boxes)
    boxes_areas = box_area(boxes)
    unions = gt_boxes_areas[:, None] + boxes_areas
    unions = unions - intersections

    ious = intersections / (unions + 1e-8)

    return ious

# class IouSimilarity:
#     """Class to compute similarity based on Intersection over Union (IOU) metric.
#     """

#     def __init__(self, mask_val=-1):
#         self.mask_val = mask_val

#     def __call__(self, boxes_1, boxes_2, boxes_1_masks=None, boxes_2_masks=None):
#         """Compute pairwise IOU similarity between ground truth boxes and anchors.
#         Args:
#           boxes_1: a float Tensor with M or B * M boxes.
#           boxes_2: a float Tensor with N or B * N boxes, the rank must be less than
#             or equal to rank of `boxes_1`.
#           boxes_1_masks: a boolean Tensor with M or B * M boxes. Optional.
#           boxes_2_masks: a boolean Tensor with N or B * N boxes. Optional.

#         Returns:
#           A Tensor with shape [M, N] or [B, M, N] representing pairwise
#             iou scores, anchor per row and groundtruth_box per colulmn.

#         Input shape:
#           boxes_1: [N, 4], or [B, N, 4]
#           boxes_2: [M, 4], or [B, M, 4]
#           boxes_1_masks: [N, 1], or [B, N, 1]
#           boxes_2_masks: [M, 1], or [B, M, 1]

#         Output shape:
#           [M, N], or [B, M, N]
#         """
#         boxes_1_rank = boxes_1.ndim
#         boxes_2_rank = boxes_2.ndim
#         if boxes_1_rank < 2 or boxes_1_rank > 3:
#             raise ValueError('`groudtruth_boxes` must be rank 2 or 3, got {}'.format(boxes_1_rank))
#         if boxes_2_rank < 2 or boxes_2_rank > 3:
#             raise ValueError('`anchors` must be rank 2 or 3, got {}'.format(boxes_2_rank))
#         if boxes_1_rank < boxes_2_rank:
#             raise ValueError('`groundtruth_boxes` is unbatched while `anchors` is '
#                             'batched is not a valid use case, got groundtruth_box '
#                             'rank {}, and anchors rank {}'.format(boxes_1_rank, boxes_2_rank))

#         result = iou(boxes_1, boxes_2)
#         if boxes_1_masks is None and boxes_2_masks is None:
#             return result
        
#         # masking off ious that are from invalid boxes
#         background_mask = None
#         # mask_val_t = np.cast(self.mask_val, result.dtype) * tf.ones_like(result)
#         perm = [1, 0] if boxes_2_rank == 2 else [0, 2, 1]
#         if boxes_1_masks is not None and boxes_2_masks is not None:
#             background_mask = np.logical_or(boxes_1_masks, np.transpose(boxes_2_masks, perm))
#         elif boxes_1_masks is not None:
#             background_mask = boxes_1_masks
#         else:
#             background_mask = np.logical_or(
#                 np.zeros(boxes_2.shape[:-1], dtype=bool),
#                 np.transpose(boxes_2_masks, perm))

#         return np.where(background_mask, self.mask_val, result)
