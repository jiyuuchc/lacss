from __future__ import annotations

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

def _resize_3d(img, target_size, **kwargs):
    assert img.ndim == 4
    assert len(target_size) == 3
    img = tf.image.resize(img, target_size[1:], **kwargs)
    img = tf.transpose(img, (2,0,1,3))
    img = tf.image.resize(img, target_size[:2], **kwargs)
    img = tf.transpose(img, (1,2,0,3))
    return img

def resize_img(img, target_size, **kwargs):
    if img.ndim == 3:
        return tf.image.resize(img, target_size, **kwargs)
    else:
        return _resize_3d(img, target_size, **kwargs)


def resize(inputs: dict, *, target_size: tuple[int, int], p: float = 1.0) -> dict:
    """Resize image and labels

    Args:
        inputs: data from TF dataset. Affected elements are:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        target_size: target size
        p: probability of applying transformation
    """

    def _resize(inputs):
        img = inputs['image']
        img_shape = tf.convert_to_tensor(tf.shape(img[..., 0]))
        target_sz = tf.convert_to_tensor(target_size)
        scalings = tf.cast(target_sz, tf.float32) / tf.cast(img_shape, tf.float32)

        inputs['image'] = resize_img(img, target_size, antialias=True)

        if "image_mask" in inputs:
            image_mask = inputs["image_mask"][..., None]  # need a channel dimension
            image_mask = resize_img(image_mask, target_size)
            inputs["image_mask"] = tf.squeeze(image_mask, axis=-1)

        if "centroids" in inputs:
            inputs["centroids"] = tf.cast(inputs["centroids"], dtype=tf.float32) * scalings

        if "bboxes" in inputs:
            inputs["bboxes"] = tf.cast(inputs["bboxes"], dtype=tf.float32) * tf.concat([scalings, scalings], axis=0)

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
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
        inputs: data from TF dataset. Affected elements are:

            * image
            * image_mask
            * centroids

    Keyword Args:
        scaling: range of scale, e.g, [0.8, 1.5].
        keep_aspect_ratio: Whether to scale x/y the same amount.
        p: probability of applying transformation

    """

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        min_scale, max_scale = scaling

        img_shape = tf.convert_to_tensor(tf.shape(inputs['image'][..., 0]))
        dim = inputs['image'].ndim - 1

        H, W = _image_size(inputs["image"])

        if keep_aspect_ratio:
            scalings = tf.random.uniform([1], min_scale, max_scale)
            # scaling_y = tf.random.uniform([], min_scale, max_scale)
            # scaling_x = scaling_y
        else:
            scalings = tf.random.uniform([dim], min_scale, max_scale)

        target_size = tf.cast(tf.round(img_shape * scaling), tf.int32)

        return resize(inputs, target_size=target_size)


def crop_to_roi(
    inputs: dict,
    *,
    roi: tuple[int, ...],
    area_ratio_threshold: float = 1.0,
    clip_boxes: bool = True,
    p: float = 1.0,
) -> dict:
    """Crop image to bounding-box ROI

    Args:
        inputs: data from TF dataset. Affected elements are:

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
        dim = inputs['image'].ndim - 1
        roi_ = tf.convert_to_tensor(roi)
        p0 = tf.cast(roi_[:dim], tf.float32)
        p1 = tf.cast(roi_[dim:], tf.float32)
        span = p1 - p0

        slices = [slice(int(p0[k]), int(p1[k])) for k in range(dim)]
        inputs['image'] = inputs['image'].__getitem__(slices)

        if "image_mask" in inputs:
            inputs["image_mask"] = inputs["image_mask"].__getitem__(slices)
        
        boolean_mask = _init_boolean_mask(inputs)
        if boolean_mask is None:
            return inputs

        if "centroids" in inputs:
            # ctrds = tf.cast(inputs["centroids"], tf.float32) - tf.cast(p0, tf.float32)
            ctrds = tf.cast(inputs["centroids"], tf.float32)
            boolean_mask &= tf.reduce_all(ctrds > p0, axis=-1) 
            boolean_mask &= tf.reduce_all(ctrds < p1, axis=-1)

        if "bboxes" in inputs:
            bboxes = tf.cast(inputs["bboxes"], tf.float32)
            bboxes = bboxes - tf.concat([p0, p0], axis=0)
            areas = tf.reduce_prod(bboxes[:, dim:] - bboxes[:, :dim], axis=-1)

            clipped_bboxes = tf.minimum(bboxes, tf.concat([span, span], axis=0))
            clipped_bboxes = tf.maximum(clipped_bboxes, 0)
            clipped_areas = tf.reduce_prod(clipped_bboxes[:, dim:] - clipped_bboxes[:, :dim], axis=-1)

            boolean_mask &= (clipped_areas / areas) >= area_ratio_threshold

        # so that TF won't complain boolean_mask has no shape
        boolean_mask = tf.ensure_shape(boolean_mask, [None])

        if "centroids" in inputs:
            inputs["centroids"] = tf.boolean_mask(ctrds, boolean_mask) - p0

        if "bboxes" in inputs:
            if clip_boxes:
                bboxes = tf.boolean_mask(bboxes, boolean_mask)
                inputs["bboxes"] = tf.boolean_mask(clipped_bboxes, boolean_mask)
            else:
                inputs["bboxes"] = tf.boolean_mask(bboxes, boolean_mask)

            if "masks" in inputs:
                if clip_boxes:
                    mask_shape = inputs["masks"][0].shape

                    bboxe_p0s = bboxes[:, :dim]
                    bboxe_szs = bboxes[:, dim:] - bboxes[:, :dim]

                    new_bbox = inputs["bboxes"] - tf.concat([bboxe_p0s, bboxe_p0s], axis=-1)
                    new_bbox = new_bbox / tf.concat([bboxe_szs, bboxe_szs], axis=-1)
                    new_bbox = new_bbox * tf.cast(
                        mask_shape + mask_shape, new_bbox.dtype
                    )
                    inputs["masks"] = _crop_and_resize(
                        tf.boolean_mask(inputs["masks"], boolean_mask),
                        new_bbox, mask_shape
                    )
                else:
                    mask_shape = inputs['masks'].shape
                    inputs["masks"] = tf.ensure_shape(
                        tf.boolean_mask(inputs["masks"], boolean_mask),
                        mask_shape,
                    )

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        return _crop_to_roi(inputs)


def random_crop(
    inputs: dict,
    *,
    target_size: int|tuple[int, ...],
    area_ratio_threshold: float = 1.0,
    clip_boxes: bool = True,
    p: float = 1.0,
) -> dict:
    """Random crop to a set target size

    Args:
        inputs: data from TF dataset. Affected elements are:

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

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        dim = inputs['image'].ndim - 1
        img_shape = tf.shape(inputs['image'])[:-1]
        target_size = tf.broadcast_to(
            tf.convert_to_tensor(target_size, dtype=tf.int32),
            [dim],
        )

        p0 = tf.random.uniform([dim]) * tf.cast(img_shape - target_size, tf.float32)
        p0 = tf.cast(p0, tf.int32)
        roi = tf.concat([p0, p0 + target_size], axis=0)
        return crop_to_roi(
            inputs, 
            roi=roi,
            area_ratio_threshold=area_ratio_threshold,
            clip_boxes=clip_boxes,
        )


def pad(
    inputs: dict,
    *,
    paddings: int | tuple[int, ...],
    constant_values: float = 0,
    p: float = 1.0,
) -> dict:
    """Pad image and labels.

    Args:
        inputs: data from TF dataset. Affected elements are:

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
        dim = inputs['image'].ndim - 1
        paddings_ = tf.convert_to_tensor(paddings, dtype=tf.int32)
        paddings_ = tf.broadcast_to(paddings_, (dim, 2))

        padding_values = tf.concat([paddings_, tf.zeros([1,2], dtype=tf.int32)], axis=0)
        inputs["image"] = tf.pad(inputs["image"], padding_values, constant_values=constant_values,)

        if "image_mask" in inputs:
            image_mask = inputs["image_mask"]
            if image_mask.ndim == dim:
                padding_values = paddings_
            else:
                padding_values = tf.concat(
                    [tf.zeros([1,2], dtype=tf.int32), paddings_],
                    axis=0,
                )
            inputs["image_mask"] = tf.pad(
                inputs["image_mask"],
                padding_values,
                constant_values=constant_values,
            )

        if "centroids" in inputs:
            inputs["centroids"] += tf.cast(paddings_[:, 0], tf.float32)

        if "bboxes" in inputs:
            inputs["bboxes"] += tf.cast(
                tf.concat([paddings_[:, 0], paddings_[:, 0]], axis=0), tf.float32
            )

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        return _pad(inputs)


def pad_to_size(
    inputs: dict,
    *,
    target_size: int|tuple[int, ...],
    constant_values: float = 0,
    p: float = 1.0,
) -> dict:
    """Pad image and labels to a target size. Padding is applied so that the orginal scene is centered in the output.

    Args:
        inputs: data from TF dataset. Affected elements are:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        target_size: target image size
        constant_values: the value to fill the padded area
        p: probability of applying transformation
    """

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    dim = inputs['image'].ndim -1
    img_shape = tf.shape(inputs['image'][..., 0])
    target_size = tf.broadcast_to(
        tf.convert_to_tensor(target_size, dtype=tf.int32),
        [dim],
    )

    paddings = tf.stack([
        tf.zeros([dim], dtype=tf.int32),
        target_size - img_shape    
    ], axis=-1)

    return pad(inputs, paddings=paddings, constant_values=constant_values)


def random_crop_or_pad(
    inputs: dict,
    *,
    target_size: tuple[int, int],
    constant_values: float = 0,
    area_ratio_threshold: float = 1.0,
    clip_boxes:bool = True,
    p: float = 1.0,
) -> dict:
    """Random crop or pad image to specified target_size.

    Args:
        inputs: data from TF dataset. Affected elements are:

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

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        dim = inputs['image'].ndim -1
        img_shape = tf.shape(inputs['image'][..., 0])
        target_size = tf.broadcast_to(
            tf.convert_to_tensor(target_size, dtype=tf.int32),
            [dim],
        )
        
        padsize = tf.maximum(target_size, img_shape)        

        inputs = pad_to_size(
            inputs, target_size=padsize, constant_values=constant_values
        )

        inputs = random_crop(
            inputs, target_size=target_size, area_ratio_threshold=area_ratio_threshold, clip_boxes=clip_boxes,
        )

        return inputs


def flip_left_right(inputs: dict, *, p: float = 1.0) -> dict:
    """Flip image left-right

    Args:
        inputs: data from TF dataset. Affected elements are:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        p: probability of applying transformation

    """

    def _flip_left_right(inputs):
        dim = inputs['image'].ndim -1
        W = tf.shape(inputs['image'])[-2]

        inputs["image"] = tf.image.flip_left_right(inputs["image"])

        if "image_mask" in inputs:
            image_mask = inputs["image_mask"][..., None]
            inputs["image_mask"] = tf.squeeze(
                tf.image.flip_left_right(image_mask),
                axis=-1,
            )

        sgn = tf.cast([1, -1] if dim==2 else [1, 1, -1], tf.float32)
        ofs = tf.cast([0, W] if dim==2 else [0, 0, W], tf.float32)
        if "centroids" in inputs:
            inputs["centroids"] = inputs["centroids"] * sgn + ofs

        if "bboxes" in inputs:
            inputs["bboxes"] = inputs["bboxes"] * tf.concat([sgn, sgn], axis=0) + tf.concat([ofs, ofs], axis=0)

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        return _flip_left_right(inputs)


def flip_up_down(inputs: dict, *, p: float = 1.0) -> dict:
    """Flip image up-down

    Args:
        inputs: data from TF dataset. Affected elements are:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        p: probability of applying transformation

    """

    def _flip_up_down(inputs):
        dim = inputs['image'].ndim -1
        H = tf.shape(inputs['image'])[-3]

        inputs["image"] = tf.image.flip_up_down(inputs["image"])

        if "image_mask" in inputs:
            image_mask = inputs["image_mask"][..., None]
            inputs["image_mask"] = tf.squeeze(
                tf.image.flip_up_down(image_mask),
                axis=-1,
            )

        sgn = tf.cast([-1, 1] if dim==2 else [1, -1, 1], tf.float32)
        ofs = tf.cast([H, 0] if dim==2 else [0, H, 0], tf.float32)
        if "centroids" in inputs:
            inputs["centroids"] = inputs["centroids"] * sgn + ofs

        if "bboxes" in inputs:
            inputs["bboxes"] = inputs["bboxes"] * tf.concat([sgn, sgn], axis=0) + tf.concat([ofs, ofs], axis=0)

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        return _flip_up_down(inputs)

def flip_top_bottom(inputs: dict, *, p: float = 1.0) -> dict:
    """Flip image up-down

    Args:
        inputs: data from TF dataset. Affected elements are:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        p: probability of applying transformation

    """
    def _flip_top_bottom(inputs):
        dim = inputs['image'].ndim -1
        assert dim == 3

        D = tf.shape(inputs['image'])[-4]

        inputs["image"] = tf.image.flip_up_down(inputs["image"])

        if "image_mask" in inputs:
            image_mask = inputs["image_mask"][..., None]
            inputs["image_mask"] = tf.squeeze(
                tf.image.flip_up_down(image_mask),
                axis=-1,
            )

        sgn = tf.cast([-1, 1, 1], tf.float32)
        ofs = tf.cast([D, 0, 0], tf.float32)
        if "centroids" in inputs:
            inputs["centroids"] = inputs["centroids"] * sgn + ofs

        if "bboxes" in inputs:
            inputs["bboxes"] = inputs["bboxes"] * tf.concat([sgn, sgn], axis=0) + tf.concat([ofs, ofs], axis=0)

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs

    else:
        return _flip_top_bottom(inputs)


def transpose(inputs, *, axes, p=1):
    def _tranpose(inputs):
        axes_ = tf.convert_to_tensor(axes, dtype=tf.int32)
        dim = inputs['image'].ndim -1

        inputs['image'] = tf.transpose(
            inputs['image'], 
            tf.concat([axes_, [dim]], axis=0),
        )
        if "image_mask" in inputs:
            inputs['image_mask'] = tf.transpose(inputs['image_mask'], axes_)

        if "masks" in inputs:
            inputs['masks'] = tf.transpose(
                inputs['masks'], 
                tf.concat([[0], axes_ + 1], axis=0),
            )

        if "centroids" in inputs:
            inputs["centroids"] = tf.gather(inputs["centroids"], axes_, axis=1)

        if "bboxes" in inputs:
            inputs["bboxes"] = tf.gather(
                inputs["bboxes"], 
                tf.concat([axes_, axes_+dim], axis=0), 
                axis=1,
            )

        return inputs

    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs
    else:
        return _tranpose(inputs)
    

def mosaic(inputs):
    img = inputs['image']
    H = tf.shape(img)[-3]
    W = tf.shape(img)[-2]
    img = tf.concat([
        tf.concat([img[0], img[1]], axis=-2),
        tf.concat([img[2], img[3]], axis=-2),
    ], axis=-3)
    inputs['image']=img
    
    ofs = tf.cast([
        [0,0],[0,W],[H, 0],[H, W]
    ], dtype=inputs['centroids'].dtype)
    ofs = tf.reshape(ofs, [4, 1, 2])
    locs = inputs['centroids'] + ofs
    inputs['centroids'] = locs.merge_dims(0,1)
    
    if "bboxes" in inputs:
        ofs_box = tf.concat([ofs, ofs], axis=-1)
        box = inputs['bboxes'] + ofs_box
        inputs['bboxes'] = box.merge_dims(0,1)
    
    if "masks" in inputs:
        inputs['masks'] = inputs['masks'].merge_dims(0,1)
    
    return inputs
    

@tf.py_function(Tout=tf.float32)
def _py_cutout(image, bbox, size, n):
    import numpy as np
    image = image.numpy()
    bbox = bbox.numpy()
    size = int(size)
    n = int(n)
    img_shape = image.shape[:-1]
    size = np.array(size).astype(int)

    sel = np.zeros([bbox.shape[0]], dtype=int)
    sel[:n] = 1
    np.random.shuffle(sel)
    sel_boxes = bbox[sel > 0]
    dim = sel_boxes.shape[-1]//2
    mins = np.minimum(sel_boxes[:,:dim], sel_boxes[:,dim:])
    maxs = np.maximum(sel_boxes[:,:dim], sel_boxes[:,dim:])
    c0s = np.random.randint(mins-size, maxs+size)

    for c0 in c0s:
        s = tuple(slice(max(0, a), min(c, b)) for a, b, c in zip(c0, c0+size, img_shape))
        image.__setitem__(s, 0)

    return image


def cutout(inputs, *, size:int=10, n:int=50, p:float=1.0):
    def _cutout(inputs):
        assert "bboxes" in inputs
        image = _py_cutout(inputs['image'], inputs['bboxes'], size, n)
        inputs['image'] = image
        return inputs
        
    if p < 1.0 and tf.random.uniform([]) >= p:
        return inputs
    else:
        return _cutout(inputs)
