import jax

from .boxes import box_iou_similarity
from .patches import bboxes_of_patches

jnp = jax.numpy


class MaskIndices:
    _row_lengths = None
    _row_starts = None

    def __init__(self, data):
        self.data = data

    def _compute_row_lengths(self):
        diff_ids = self.data[1:, -1] - self.data[:-1, -1]
        jump_locs = jnp.argwhere(diff_ids)
        jump_locs = jump_locs.repeat(diff_ids[jump_locs]) + 1
        jump_locs = jnp.concatenate(
            [jnp.array([0]), jump_locs, jnp.array([self.data.shape[0]])]
        )

        self._row_starts = jump_locs[:-1]
        self._row_lengths = jump_locs[1:] - jump_locs[:-1]

    @property
    def row_lengths(self):
        if self._row_lengths is None:
            self._compute_row_lengths()
        return self._row_lengths

    @property
    def row_starts(self):
        if self._row_starts is None:
            self._compute_row_lengths()
        return self._row_starts

    @property
    def n_masks(self):
        return len(self.row_lengths)

    def gather(self, indices):
        sel_starts = self.row_starts[indices]
        sel_repeats = self.row_lengths[indices]
        new_data = jnp.concatenate(
            [self.data[s : s + l, :2] for s, l in zip(sel_starts, sel_repeats)]
        )
        new_ids = jnp.arange(sel_starts.size).repeat(sel_repeats)

        new_masks = MaskIndices(jnp.append(new_data, new_ids[:, None], axis=-1))
        new_masks._row_lengths = sel_repeats
        new_masks._row_starts = jnp.insert(sel_repeats[:-1].cumsum(), 0, 0)

        return new_masks

    def to_patches(self, y0x0, patch_size: int):
        y0x0_repeated = y0x0.repeat(self.row_lengths, axis=0)
        shifted_data = self.data.at[:, :2].add(-y0x0_repeated)
        patches = jnp.zeros([self.n_masks, patch_size, patch_size], dtype=int)
        patches = patches.at[
            shifted_data[:, 2], shifted_data[:, 0], shifted_data[:, 0]
        ].set(1)

        return patches

    def bboxes(self):
        min_y, min_x = jax.ops.segment_min(
            self.data[:, :2], self.data[:, -1]
        ).transpose()
        max_y, max_x = (
            jax.ops.segment_max(self.data[:, :2], self.data[:, -1]).transpose() + 1
        )
        bboxes = jnp.stack([min_y, min_x, max_y, max_x], axis=-1)

        bboxes = jnp.where(self.row_lengths[:, None] > 0, bboxes, -1)

        return bboxes


def mask_iou_similarity(gt_masks, pred_masks, gt_boxes, pred_boxes, patch_size):
    """Compute mask_iou matrix.
    The method is to compute box iou matrix first and update only non-zero
    entries with the accurate mask_iou

    Args:
        gt_masks: [N, 3]  ground truth mask indices data. format: [yy, xx, mask_id]
        pred_masks: [M, 3] prediction mask indices data. format: [yy, xx, mask_id]
        gt_boxes: [N, 4]  y0x0y1x1 format
        pred_boxes: [M, 4] y0x0y1x1 format
        patch_size: int constant
    Outputs:
        similarity_matrix: [n_pred_instances, n_gt_instances]
    """
    gt_masks = MaskIndices(gt_masks)
    pred_masks = MaskIndices(pred_masks)

    if gt_boxes is None:
        gt_boxes = gt_masks.bboxes()
    if pred_boxes is None:
        pred_boxes = pred_masks.bboxes()

    sm = box_iou_similarity(pred_boxes, gt_boxes)

    if (sm > 0).any():
        pred_mask_ids, gt_mask_ids = jnp.argwhere(sm > 0).transpose()

        sub_gt_masks = gt_masks.gather(gt_mask_ids)
        sub_pred_masks = pred_masks.gather(pred_mask_ids)

        y0x0 = pred_boxes[pred_mask_ids, :2]
        sub_gt_patches = sub_gt_masks.to_patches(y0x0, patch_size)
        sub_pred_patches = sub_pred_masks.to_patches(y0x0, patch_size)

        area = sub_gt_masks.row_lengths + sub_pred_masks.row_lengths
        intersects = (sub_gt_patches * sub_pred_patches).sum(axis=(1, 2))
        iou = intersects / (area - intersects + 1e-8)

        sm = sm.at[pred_mask_ids, gt_mask_ids].set(iou)

    return sm


def has_box_intersection(gt_box, pred_box):
    gt_box = gt_box.astype(jnp.float32)
    pred_box = pred_box.astype(jnp.float32)

    y_min1, x_min1, y_max1, x_max1 = jnp.split(gt_box, 4, -1)
    y_min2, x_min2, y_max2, x_max2 = jnp.split(pred_box, 4, -1)

    y_min_max = jnp.minimum(y_max1, jnp.transpose(y_max2))
    y_max_min = jnp.maximum(y_min1, jnp.transpose(y_min2))
    x_min_max = jnp.minimum(x_max1, jnp.transpose(x_max2))
    x_max_min = jnp.maximum(x_min1, jnp.transpose(x_min2))

    intersect_heights = y_min_max - y_max_min
    intersect_widths = x_min_max - x_max_min

    return (intersect_heights > 0) & (intersect_widths > 0)


def mask_intersects(gt_label, gt_box, pred):
    pred_box = bboxes_of_patches(pred)
    its_table = has_box_intersection(gt_box, pred_box)
    gt_ids, pred_ids = jnp.moveaxis(jnp.argwhere(its_table), -1, 0)

    pred_patches = pred["instance_mask"][pred_ids]
    pred_patches = jnp.squeeze(pred_patches, -1)
    pred_patches = pred_patches >= 0.5

    ycs = pred["instance_yc"][pred_ids]
    xcs = pred["instance_xc"][pred_ids]
    gt_patches = gt_label[ycs, xcs] == (gt_ids + 1)[:, None, None]

    its = jnp.count_nonzero(pred_patches & gt_patches, axis=(1, 2))

    intersects = jnp.zeros_like(its_table, dtype=jnp.float32)
    intersects = intersects.at[gt_ids, pred_ids].set(its)

    return intersects


def mask_ious(gt_label, gt_box, pred):
    """Compute mask_iou matrix
    Args:
        gt_label: [H,W]
        gt_box: [N, 4]
        pred: dict. model output without batch dimension.
    Outputs:
        similarity_matrix: Tensor (float32) [n_pred_masks, n_gt_masks]
    """
    n_labels = gt_label.max()
    gt_areas = jnp.count_nonzero(
        gt_label == jnp.arange(1, n_labels + 1)[:, None, None], axis=(1, 2)
    )
    pred_areas = jnp.count_nonzero(pred["instance_mask"] >= 0.5, axis=(1, 2))
    sum_areas = gt_areas[:, None] + pred_areas
    intersects = mask_intersects(gt_label, gt_box, pred)
    ious = intersects / (sum_areas - intersects + 1e-8)

    return jnp.transpose(ious)
