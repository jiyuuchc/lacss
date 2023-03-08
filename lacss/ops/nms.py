from typing import Callable

import jax

from .boxes import box_iou_similarity
from .locations import distance_similarity

jnp = jax.numpy

NMS_TILE_SIZE = 512


def _suppress(boxes, mask):
    return jnp.where(mask[:, None], -1, boxes)


def _suppression_loop_body(inputs):
    """Process boxes in the range [idx*NMS_TILE_SIZE, (idx+1)*NMS_TILE_SIZE).
    Args:
        inputs: tuple
            idx: current slice
            boxes: a tensor with a shape of [N, 4].
            num_selected: number of selected boxes so far
            threshold: float
    Returns:
        boxes: updated
        num_selected: updated
    """

    # Iterates over tiles that can possibly suppress the current tile.

    idx, boxes, num_selected, threshold = inputs

    box_slice = boxes[idx]
    similarity_func = (
        box_iou_similarity if boxes.shape[-1] == 4 else distance_similarity
    )

    def _for_loop_func(idx, slice):
        iou = similarity_func(boxes[idx], slice)
        suppressed = jnp.any(iou >= threshold, axis=0)
        return _suppress(slice, suppressed)

    box_slice = jax.lax.fori_loop(0, idx, _for_loop_func, box_slice)

    # Iterates over the current tile to compute self-suppression.
    iou = similarity_func(box_slice, box_slice)
    mask = jnp.arange(NMS_TILE_SIZE).reshape([1, -1]) > jnp.arange(
        NMS_TILE_SIZE
    ).reshape([-1, 1])
    mask = mask & (iou > threshold)

    def _while_loop_func(inputs):
        mask, cnt = inputs
        cnt = jnp.count_nonzero(mask)
        can_suppress_others = ~mask.any(axis=0)
        suppressed = (mask & can_suppress_others[:, None]).any(axis=0)
        mask = mask & ~suppressed[:, None]
        # suppressed = mask.at[can_suppress_others, :].any(axis=0)
        # mask = mask.at[suppressed, :].set(False)
        return mask, cnt

    def _while_cond_func(inputs):
        mask, cnt = inputs
        return jnp.count_nonzero(mask) != cnt

    mask, _ = jax.lax.while_loop(_while_cond_func, _while_loop_func, (mask, 0))
    box_slice = _suppress(box_slice, mask.any(axis=0))

    # Uses box_slice to update the input boxes.
    boxes = boxes.at[idx].set(box_slice)

    # output_size.
    num_selected += jnp.count_nonzero((box_slice >= 0).any(axis=-1))

    return boxes, num_selected


def sorted_non_max_suppression(
    scores,
    boxes,
    max_output_size: int,
    threshold: float = 0.5,
    min_score: float = 0,
):
    """non-maximum suppression.
    Assumption:
        * The boxes are sorted by scores
    The overal design of the algorithm is to handle boxes tile-by-tile:
    Args:
        scores: [N]
        boxes: [N, C]  C=4 for boxes, C=2 for locations
        max_output_size: a positive scalar integer
        threshold: a scalar float, can be negative
        min_score: min score to be selected, default 0
    Returns:
        nms_scores: [M].  M = max_output_size
        nms_proposals: [M, C].
    """
    # preprocessing
    c = boxes.shape[-1]
    if c != 2 and c != 4:
        raise ValueError(f"boxes should be Nx4 or Nx2, got Nx{c}")
    # if similarity_func is None:
    #     similarity_func = box_iou_similarity if c == 4 else distance_similarity

    if max_output_size <= 0:
        max_output_size = boxes.shape[0]

    # pad_to_multiply_of(tile_size)
    num_boxes = boxes.shape[0]
    pad = NMS_TILE_SIZE - 1 - (num_boxes - 1) % NMS_TILE_SIZE
    boxes = jnp.pad(boxes, [[0, pad], [0, 0]], constant_values=-1)
    scores = jnp.pad(scores, [[0, pad]], constant_values=-1)
    num_boxes += pad
    boxes = boxes.reshape(-1, NMS_TILE_SIZE, c)

    def _trivial_suppress_all(inputs):
        (
            idx,
            boxes,
            num_outputs,
            _,
        ) = inputs
        boxes = boxes.at[idx].set(-1)
        return boxes, num_outputs

    # process all tiles until generating enough output
    num_selected = 0
    for idx in range(num_boxes // NMS_TILE_SIZE):
        boxes, num_selected = jax.lax.cond(
            (scores[idx * NMS_TILE_SIZE] >= min_score)
            & (num_selected < max_output_size),
            _suppression_loop_body,
            _trivial_suppress_all,
            (idx, boxes, num_selected, threshold),
        )

    # reshape boxes back
    boxes = boxes.reshape(-1, c)

    # remove suppressed boxes
    selected = jnp.argwhere(
        (boxes >= 0).any(axis=-1), size=max_output_size, fill_value=-1
    ).squeeze(-1)
    scores = jnp.where(selected >= 0, scores[selected], -1.0)
    boxes = jnp.where(selected[:, None] >= 0, boxes[selected, :], -1.0)

    # remove low score
    selected = scores >= min_score
    scores = jnp.where(selected, scores, -1)
    boxes = jnp.where(selected[:, None], boxes, -1)

    return scores, boxes
