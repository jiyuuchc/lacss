from __future__ import annotations

from random import random
import numpy as np

def _uniform_rand(min_v, max_v):
    return random() * (max_v - min_v) + min_v

def _init_boolean_mask(inputs):
    if "centroids" in inputs:
        return np.ones([len(inputs["centroids"])], dtype=bool)
    elif "bboxes" in inputs:
        return np.ones([len(inputs["bboxes"])], dtype=bool)
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
    H = img.shape[-3]
    W = img.shape[-2]

    return (H, W)

#FIXME try a np implementation
def _crop_and_resize(masks, boxes, target_shape):
    import jax
    from lacss.ops import sub_pixel_crop_and_resize
    return np.asarray(jax.vmap(
        sub_pixel_crop_and_resize,
        in_axes=(0, 0, None),
    )(masks, boxes, target_shape))

def resize(inputs: dict, *, target_size: tuple[int, int], p: float = 1.0) -> dict:
    """Resize image and labels

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        target_size: target size
        p: probability of applying transformation
    """

    def _resize(inputs):
        from skimage.transform import resize
        H, W = _image_size(inputs["image"])

        target_y, target_x = target_size

        scaling_y = target_y / H
        scaling_x = target_x / W

        inputs["image"] = resize(
            inputs["image"],
            [target_y, target_x],
            antialias=True,
        )

        if "image_mask" in inputs:
            image_mask = inputs["image_mask"] / inputs["image_mask"].max() 
            image_mask = resize(image_mask, [target_y, target_x])
            image_mask = (image_mask >= 0.5).astype(inputs["image_mask"].dtype)
            inputs["image_mask"] = image_mask

        if "centroids" in inputs:
            inputs["centroids"] *= np.asarray([scaling_y, scaling_x])

        if "bboxes" in inputs:
            inputs["bboxes"] *= np.asarray([scaling_y, scaling_x, scaling_y, scaling_x])
            
        return inputs

    if p < 1.0 and random() >= p:
        return inputs

    else:
        return _resize(inputs)


def random_resize(
    inputs: dict,
    *,
    scaling: float | tuple[float, float],
    keep_aspect_ratio: bool = False,
    p: float = 1.0,
) -> dict:
    """Resize image by a random amount

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids

    Keyword Args:
        scaling: range of scale, e.g, [0.8, 1.5].
        keep_aspect_ratio: Whether to scale x/y the same amount.
        p: probability of applying transformation

    """

    if p < 1.0 and random() >= p:
        return inputs

    else:
        min_scale, max_scale = scaling
        H, W = _image_size(inputs["image"])

        if keep_aspect_ratio:
            scaling_x = scaling_y = _uniform_rand(min_scale, max_scale)
        else:
            scaling_y = _uniform_rand(min_scale, max_scale)
            scaling_x = _uniform_rand(min_scale, max_scale)

        target_h = scaling_y * H
        target_w = scaling_x * W

        return resize(inputs, target_size=(target_h, target_w))


def crop_to_roi(
    inputs: dict,
    *,
    roi: tuple[int, int, int, int],
    area_ratio_threshold: float = 1.0,
    p: float = 1.0,
) -> dict:
    """Crop image to bounding-box ROI

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes
            * masks

    Keyword Args:
        roi: Rectangle roi in yxyx format
        area_ratio_threshold: remove instances if the bbox's relative remaining area is below this threshold
        p: probability of applying transformation

    """

    def _crop_to_roi(inputs):
        y0, x0, y1, x1 = roi
        roi_h = int(y1 - y0)
        roi_w = int(x1 - x0)

        inputs["image"] = inputs["image"][y0:y1, x0:x1, :]

        if "image_mask" in inputs:
            inputs["image_mask"] = inputs["image_mask"][y0:y1, x0:x1]

        boolean_mask = _init_boolean_mask(inputs)

        if boolean_mask is None:
            return inputs

        if "centroids" in inputs:
            ctrds = inputs["centroids"] - [y0, x0]
            boolean_mask &= (ctrds > 0).all(axis=-1) & (ctrds < [roi_h, roi_w]).all(axis=-1)

        if "bboxes" in inputs:
            bboxes = inputs["bboxes"] - [y0, x0, y0, x0]
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

            bboxes = np.clip(bboxes, 0, [roi_h, roi_w, roi_h, roi_w])
            new_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

            boolean_mask &= (new_areas / areas) >= area_ratio_threshold

        if "centroids" in inputs:
            inputs["centroids"] = ctrds[boolean_mask]

        if "bboxes" in inputs:
            orig_bboxes = inputs["bboxes"]
            inputs["bboxes"] = bboxes[boolean_mask]

            if "masks" in inputs:
                target_shape = inputs["masks"].shape[1:3]

                orig_bboxes = orig_bboxes[boolean_mask]
                orig_y0 = orig_bboxes[:, 0] - y0
                orig_x0 = orig_bboxes[:, 1] - x0
                orig_h = orig_bboxes[:, 2] - orig_bboxes[:, 0]
                orig_w = orig_bboxes[:, 3] - orig_bboxes[:, 1]

                new_bbox = inputs["bboxes"] - [orig_y0, orig_x0, orig_y0, orig_x0]
                new_bbox = new_bbox / [orig_h, orig_w, orig_h, orig_w]
                new_bbox = new_bbox * (target_shape + target_shape)

                inputs["masks"] = _crop_and_resize(
                    inputs["masks"][boolean_mask],
                    new_bbox,
                    target_shape,
                )

        return inputs

    if p < 1.0 and random() >= p:
        return inputs

    else:
        return _crop_to_roi(inputs)


def random_crop(
    inputs: dict,
    *,
    target_size: tuple[int, int],
    area_ratio_threshold: float = 1.0,
    p: float = 1.0,
) -> dict:
    """Random crop to a set target size

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes
            * masks

    Keyword Args:
        target_size: the target size
        area_ratio_threshold: remove instances if the bbox's relative remaining area is below this threshold
        p: probability of applying transformation
    """

    H, W = _image_size(inputs["image"])

    target_h, target_w = _value_pair(target_size)

    if p < 1.0 and random() >= p:
        return inputs

    else:
        y0 = int(random() * (H - target_h))
        x0 = int(random() * (W - target_w))

        return crop_to_roi(
            inputs,
            roi=(y0, x0, y0 + target_h, x0 + target_w),
            area_ratio_threshold=area_ratio_threshold,
        )


def pad(
    inputs: dict,
    *,
    paddings: int | tuple[int, int],
    constant_values: float = 0,
    p: float = 1.0,
) -> dict:
    """Pad image and labels.

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        paddings: either a tuple or a single value. If latter, use the same padding for both x and y axis
        constant_values: the value to fill the padded area
        p: probability of applying transformation
    """

    def _pad(inputs):
        padding_y, padding_x = _value_pair(paddings)
        padding_y0, padding_y1 = _value_pair(padding_y)
        padding_x0, padding_x1 = _value_pair(padding_x)

        inputs["image"] = np.pad(
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
            inputs["image_mask"] = np.pad(
                inputs["image_mask"],
                padding_values,
                constant_values=constant_values,
            )

        if "centroids" in inputs:
            inputs["centroids"] += np.asarray([padding_y0, padding_x0])

        if "bboxes" in inputs:
            inputs["bboxes"] += np.asarray(
                [padding_y0, padding_x0, padding_y0, padding_x0], 
            )

        return inputs

    if p < 1.0 and random() >= p:
        return inputs

    else:
        return _pad(inputs)


def pad_to_size(
    inputs: dict,
    *,
    target_size: tuple[int, int],
    constant_values: float = 0,
    p: float = 1.0,
) -> dict:
    """Pad image and labels to a target size. Padding is applied so that the orginal scene is centered in the output.

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        target_size: target image size
        constant_values: the value to fill the padded area
        p: probability of applying transformation
    """

    if p < 1.0 and random() >= p:
        return inputs

    H, W = _image_size(inputs["image"])
    target_h, target_w = _value_pair(target_size)

    padding_h = target_h - H
    padding_h = [padding_h // 2, padding_h - padding_h // 2]
    padding_w = target_w - W
    padding_w = [padding_w // 2, padding_w - padding_w // 2]

    return pad(inputs, paddings=[padding_h, padding_w], constant_values=constant_values)


def random_crop_or_pad(
    inputs: dict,
    *,
    target_size: tuple[int, int],
    constant_values: float = 0,
    area_ratio_threshold: float = 1.0,
    p: float = 1.0,
) -> dict:
    """Random crop or pad image to specified target_size.

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes
            * masks

    Keyword Args:
        target_size: target size
        constant_values: the value to fill the padded area
        area_ratio_threshold: remove instances if the bbox's relative remaining area is below this threshold
        p: probability of applying transformation
    """

    if p < 1.0 and random() >= p:
        return inputs

    else:
        H, W = _image_size(inputs["image"])

        target_h, target_w = _value_pair(target_size)

        h = max(H, target_h)
        w = max(W, target_w)

        inputs = pad_to_size(
            inputs, target_size=(h, w), constant_values=constant_values
        )

        inputs = random_crop(
            inputs, target_size=target_size, area_ratio_threshold=area_ratio_threshold
        )

        return inputs


def flip_left_right(inputs: dict, *, p: float = 1.0) -> dict:
    """Flip image left-right

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        p: probability of applying transformation

    """

    def _flip_left_right(inputs):
        H, W = _image_size(inputs["image"])

        inputs["image"] = inputs["image"][:, ::-1, :]

        if "image_mask" in inputs:
            inputs["image_mask"] = inputs["image_mask"][:, ::-1]

        if "centroids" in inputs:
            inputs["centroids"] = inputs["centroids"] * [1, -1] + [0, W]

        #FIXME reformat bboxes so no negative length
        if "bboxes" in inputs:
            inputs["bboxes"] = inputs["bboxes"] * [1, -1, 1, -1] + [0, W, 0, W]

        # if "masks" in inputs:
        #     inputs["masks"] = inputs["masks"][:, :, ::-1, :]

        return inputs

    if p < 1.0 and random() >= p:
        return inputs

    else:
        return _flip_left_right(inputs)


def flip_up_down(inputs: dict, *, p: float = 1.0) -> dict:
    """Flip image up-down

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        p: probability of applying transformation

    """

    def _flip_up_down(inputs):
        H, W = _image_size(inputs["image"])

        inputs["image"] = inputs["image"][::-1, :, :]

        if "image_mask" in inputs:
            inputs["image_mask"] = inputs["image_mask"][::-1, :]

        if "centroids" in inputs:
            inputs["centroids"] = inputs["centroids"] * [-1, 1] + [H, 0]

        #FIXME reformat bboxes so no negative length
        if "bboxes" in inputs:
            inputs["bboxes"] = inputs["bboxes"] * [-1, 1, -1, 1] + [H, 0, H, 0]

        # if "masks" in inputs:
        #     inputs["masks"] = inputs["masks"][:, ::-1, :, :]

        return inputs

    if p < 1.0 and random() >= p:
        return inputs

    else:
        return _flip_up_down(inputs)
