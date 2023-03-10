import jax

jnp = jax.numpy


def box_area(box):
    """Computes area of boxes.
    Args:
      box: a float Tensor with [..., N, 4].

    Returns:
      a float Tensor with [..., N]
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
      gt_boxes: a float Tensor with [..., N, 4].
      boxes: a float Tensor with [..., M, 4].

    Returns:
      a Tensor with shape [..., N, M] representing pairwise iou scores.
    """
    intersections = box_intersection(gt_boxes, boxes)
    gt_boxes_areas = box_area(gt_boxes)
    boxes_areas = box_area(boxes)
    unions = gt_boxes_areas[:, None] + boxes_areas - intersections

    ious = intersections / (unions + 1e-6)

    return ious
