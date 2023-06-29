""" A set of augmentation functions involving shape change
"""

import tensorflow as tf

from .generator import _crop_and_resize


def _init_boolean_mask(inputs):
    if "centroids" in inputs:
        return tf.ones([len(inputs["centroids"])], dtype=tf.bool)
    elif "bboxes" in inputs:
        return tf.ones([len(inputs["bboxes"])], dtype=tf.bool)
    else:
        return None


def _value_pair(x):
    try:
        a, b = x
    except:
        a = x
        b = x

    return a, b


def _image_size(img):
    H = tf.shape(img)[-3]
    W = tf.shape(img)[-2]

    return (H, W)


def resize(inputs, *, target_size, p=1.0):
    """Resize image and labels"""

    def _resize(inputs):
        H, W = _image_size(inputs["image"])
        target_y, target_x = target_size

        scaling_y = target_y / tf.cast(H, tf.float32)
        scaling_x = target_x / tf.cast(W, tf.float32)

        inputs["image"] = tf.image.resize(
            inputs["image"],
            [target_y, target_x],
            antialias=True,
        )

        if "image_mask" in inputs:
            image_mask = (inputs["image_mask"][..., None],)  # need a channel dimension
            image_mask = tf.cast(
                tf.image.resize(
                    inputs["image_mask"][..., None],  # need a channel dimension
                    [target_y, target_x],
                )
                >= 0.5,
                inputs["image_mask"].dtype,
            )
            inputs["image_mask"] = tf.squeeze(image_mask, axis=-1)

        if "centroids" in inputs:
            inputs["centroids"] *= tf.cast([scaling_y, scaling_x], tf.float32)

        if "bboxes" in inputs:
            inputs["bboxes"] *= tf.cast(
                [scaling_y, scaling_x, scaling_y, scaling_x], tf.float32
            )

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        return _resize(inputs)


def random_resize(inputs, *, scaling, keep_aspect_ratio=False, p=1.0):
    """Random resize image by a range of scale"""

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        min_scale, max_scale = _value_pair(scaling)
        if min_scale == max_scale and max_scale < 1:
            min_scale = 1.0 - min_scale
            max_scale = 1.0 + max_scale

        H, W = _image_size(inputs["image"])

        if keep_aspect_ratio:
            scaling_y = tf.random.uniform([], min_scale, max_scale)
            scaling_x = scaling_y
        else:
            scaling_y = tf.random.uniform([], min_scale, max_scale)
            scaling_x = tf.random.uniform([], min_scale, max_scale)

        target_h = tf.round(scaling_y * tf.cast(H, tf.float32))
        target_w = tf.round(scaling_x * tf.cast(W, tf.float32))

        return resize(inputs, target_size=(target_h, target_w))


def crop_to_roi(inputs, *, roi, area_ratio_threshold=1.0, p=1.0):
    """Crop image to bounding-box ROI"""

    def _crop_to_roi(inputs):
        y0, x0, y1, x1 = roi
        roi_h = tf.cast(y1 - y0, tf.float32)
        roi_w = tf.cast(x1 - x0, tf.float32)

        inputs["image"] = inputs["image"][y0:y1, x0:x1, :]

        if "image_mask" in inputs:
            inputs["image_mask"] = inputs["image_mask"][y0:y1, x0:x1]

        boolean_mask = _init_boolean_mask(inputs)

        if boolean_mask is None:
            return inputs

        if "centroids" in inputs:
            ctrds = inputs["centroids"] - tf.cast([y0, x0], tf.float32)
            boolean_mask &= (
                tf.reduce_all(ctrds > 0, axis=-1)
                & (ctrds[:, 0] < roi_h)
                & (ctrds[:, 1] < roi_w)
            )

        if "bboxes" in inputs:
            bboxes = inputs["bboxes"] - tf.cast([y0, x0, y0, x0], tf.float32)
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

            bboxes = tf.minimum(
                bboxes, tf.cast([roi_h, roi_w, roi_h, roi_w], tf.float32)
            )
            bboxes = tf.maximum(bboxes, 0)
            new_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

            boolean_mask &= (new_areas / areas) >= area_ratio_threshold

        # so that TF won't complain boolean_mask has no shape
        boolean_mask = tf.ensure_shape(boolean_mask, [None])

        if "centroids" in inputs:
            inputs["centroids"] = tf.boolean_mask(ctrds, boolean_mask)

        if "bboxes" in inputs:
            orig_bboxes = tf.boolean_mask(inputs["bboxes"], boolean_mask)
            inputs["bboxes"] = tf.boolean_mask(bboxes, boolean_mask)

            if "masks" in inputs:
                target_shape = inputs["masks"].shape[1:3]

                orig_y0 = orig_bboxes[:, 0] - tf.cast(y0, orig_bboxes.dtype)
                orig_x0 = orig_bboxes[:, 1] - tf.cast(x0, orig_bboxes.dtype)
                orig_h = orig_bboxes[:, 2] - orig_bboxes[:, 0]
                orig_w = orig_bboxes[:, 3] - orig_bboxes[:, 1]

                new_bbox = inputs["bboxes"] - tf.stack(
                    [orig_y0, orig_x0, orig_y0, orig_x0], axis=-1
                )
                new_bbox = new_bbox / tf.stack(
                    [orig_h, orig_w, orig_h, orig_w], axis=-1
                )
                new_bbox = new_bbox * tf.cast(
                    (target_shape + target_shape), new_bbox.dtype
                )

                inputs["masks"] = _crop_and_resize(
                    tf.boolean_mask(inputs["masks"], boolean_mask),
                    new_bbox,
                    target_shape,
                )

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        return _crop_to_roi(inputs)


def random_crop(inputs, *, target_size, area_ratio_threshold=1.0, p=1.0):
    """Random crop to a set target size"""

    H, W = _image_size(inputs["image"])

    target_h, target_w = _value_pair(target_size)

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        y0 = int(tf.random.uniform([]) * tf.cast(H - target_h, tf.float32))
        x0 = int(tf.random.uniform([]) * tf.cast(W - target_w, tf.float32))

        return crop_to_roi(
            inputs,
            roi=(y0, x0, y0 + target_h, x0 + target_w),
            area_ratio_threshold=area_ratio_threshold,
        )


def pad(inputs, *, paddings, constant_values=0, p=1.0):
    """Pad image and labels."""

    def _pad(inputs):
        padding_y, padding_x = _value_pair(paddings)
        padding_y0, padding_y1 = _value_pair(padding_y)
        padding_x0, padding_x1 = _value_pair(padding_x)

        inputs["image"] = tf.pad(
            inputs["image"],
            [[padding_y0, padding_y1], [padding_x0, padding_x1], [0, 0]],
            constant_values=constant_values,
        )

        if "image_mask" in inputs:
            image_mask = inputs["image_mask"]
            if len(image_mask.shape) == 2:
                padding_values = [[padding_y0, padding_y1], [padding_x0, padding_x1]]
            else:
                padding_values = [
                    [0, 0],
                    [padding_y0, padding_y1],
                    [padding_x0, padding_x1],
                ]
            inputs["image_mask"] = tf.pad(
                inputs["image_mask"],
                padding_values,
                constant_values=constant_values,
            )

        if "centroids" in inputs:
            inputs["centroids"] += tf.cast([padding_y0, padding_x0], tf.float32)

        if "bboxes" in inputs:
            inputs["bboxes"] += tf.cast(
                [padding_y0, padding_x0, padding_y0, padding_x0], tf.float32
            )

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        return _pad(inputs)


def pad_to_size(inputs, *, target_size, constant_values=0, p=1.0):
    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    H, W = _image_size(inputs["image"])
    target_h, target_w = _value_pair(target_size)

    padding_h = target_h - H
    padding_h = [padding_h // 2, padding_h - padding_h // 2]
    padding_w = target_w - W
    padding_w = [padding_w // 2, padding_w - padding_w // 2]

    return pad(inputs, paddings=[padding_h, padding_w], constant_values=constant_values)


def random_crop_or_pad(
    inputs, *, target_size, constant_values=0, area_ratio_threshold=1.0, p=1.0
):
    """Random crop or pad image to target_size"""

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        H, W = _image_size(inputs["image"])

        target_h, target_w = _value_pair(target_size)

        h = tf.cast(tf.maximum(H, target_h), H.dtype)
        w = tf.cast(tf.maximum(W, target_w), W.dtype)

        inputs = pad_to_size(
            inputs, target_size=(h, w), constant_values=constant_values
        )

        inputs = random_crop(
            inputs, target_size=target_size, area_ratio_threshold=area_ratio_threshold
        )

        return inputs


def flip_left_right(inputs, *, p=1.0):
    def _flip_left_right(inputs):
        H, W = _image_size(inputs["image"])

        inputs["image"] = tf.image.flip_left_right(inputs["image"])

        if "image_mask" in inputs:
            image_mask = inputs["image_mask"][..., None]
            inputs["image_mask"] = tf.squeeze(
                tf.image.flip_left_right(image_mask),
                axis=-1,
            )

        if "centroids" in inputs:
            inputs["centroids"] = inputs["centroids"] * [1, -1] + [0, W]

        if "bboxes" in inputs:
            inputs["bboxes"] = inputs["bboxes"] * [1, -1, 1, -1] + [0, W, 0, W]

        # if "masks" in inputs:
        #     inputs["masks"] = inputs["masks"][:, :, ::-1, :]

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        return _flip_left_right(inputs)


def flip_up_down(inputs, *, p=1.0):
    def _flip_up_down(inputs):
        H, W = _image_size(inputs["image"])

        inputs["image"] = tf.image.flip_up_down(inputs["image"])

        if "image_mask" in inputs:
            image_mask = inputs["image_mask"][..., None]
            inputs["image_mask"] = tf.squeeze(
                tf.image.flip_up_down(image_mask),
                axis=-1,
            )

        if "centroids" in inputs:
            inputs["centroids"] = inputs["centroids"] * [-1, 1] + [H, 0]

        if "bboxes" in inputs:
            inputs["bboxes"] = inputs["bboxes"] * [-1, 1, -1, 1] + [H, 0, H, 0]

        # if "masks" in inputs:
        #     inputs["masks"] = inputs["masks"][:, ::-1, :, :]

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        return _flip_up_down(inputs)


def _bbox_from_label(label):
    import numpy as np
    from skimage.measure import regionprops

    bboxes = np.asarray([r["bbox"] for r in regionprops(label)])
    centroids = np.asarray([r["centroids"] for r in regionprops(label)])

    return bboxes, centroids


def mask_from_label(inputs, *, mask_shape=[48, 48]):
    label = inputs["label"]

    if not "bboxes" in inputs or not "centroids" in inputs:
        inputs["bboxes"], inputs["centroids"] = tf.numpy_func(
            _bbox_from_label,
            [label],
            [tf.float32, tf.float32],
            False,
        )

    n_instances = len(inputs["bboxes"])

    label_expanded = tf.cast(
        label == tf.range(1, n_instances + 1)[:, None, None],
        tf.int32,
    )

    inputs["masks"] = _crop_and_resize(
        label_expanded,
        inputs["bboxes"],
        mask_shape,
    )

    return inputs
