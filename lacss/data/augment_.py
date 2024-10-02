from __future__ import annotations

from typing import Sequence

from random import random
import numpy as np
from ..typing import ArrayLike

def _uniform_rand(min_v, max_v):
    return random() * (max_v - min_v) + min_v

def _init_boolean_mask(inputs):
    if "centroids" in inputs:
        return np.ones([len(inputs["centroids"])], dtype=bool)
    elif "bboxes" in inputs:
        return np.ones([len(inputs["bboxes"])], dtype=bool)
    else:
        return None

#FIXME try a np implementation
def _crop_and_resize(masks, boxes, target_shape):
    import jax

    from lacss.ops import sub_pixel_crop_and_resize
    return np.asarray(jax.vmap(
        sub_pixel_crop_and_resize,
        in_axes=(0, 0, None),
    )(masks, boxes, target_shape))


def _box_area(boxes):
    boxes = np.asarray(boxes)
    dim = boxes.shape[-1] // 2
    return np.abs(np.prod(boxes[..., dim:] - boxes[..., :dim], axis=-1))


def resize(inputs: dict, *, target_size: tuple[int, int]|tuple[int, int, int], use_jax:bool=False, p: float = 1.0) -> dict:
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
    def _resize_img(img, shape):
        if use_jax:
            import jax
            if img.ndim > len(shape):
                shape = list(shape) + [img.shape[-1]]
            return np.asarray(jax.image.resize(img, shape, "linear"))
        else:
            from skimage.transform import resize
            return resize(img, shape)    

    def _resize(inputs):
        scaling = np.array(target_size) / inputs["image"].shape[:-1]
        inputs["image"] = _resize_img(inputs["image"], target_size)

        if "image_mask" in inputs:
            inputs["image_mask"] = _resize_img(inputs["image_mask"], target_size)
            # if inputs["image_mask"].dtype is np.dtypes.BoolDType:
            #     image_mask = image_mask >= inputs["image_mask"].max() * 0.5
            # inputs["image_mask"] = image_mask.astype(inputs["image_mask"].dtype)

        if "centroids" in inputs:
            inputs["centroids"] = inputs["centroids"] * scaling

        if "bboxes" in inputs:
            inputs["bboxes"] = inputs["bboxes"] * np.r_[scaling, scaling]

        return inputs

    if p < 1.0 and random() >= p:
        return inputs
    if tuple(target_size) == inputs['image'].shape[:-1]:
        return inputs
    else:
        return _resize(inputs.copy())


def rescale(
    inputs: dict,
    *,
    rescale: float|Sequence[float],
    use_jax:bool=False,
    p: float = 1.0,    
):
    """Rescale image and labels 

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        rescale: rescaling factor
        p: probability of applying transformation
    """
    if p < 1.0 and random() >= p:
        return inputs

    imgsz = np.array(inputs['image'].shape[:-1])
    target_size = np.round(imgsz * rescale).astype(int)

    return resize(inputs, target_size=tuple(target_size), use_jax=use_jax)


def random_resize(
    inputs: dict,
    *,
    scaling: Sequence[float],
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
        dim = inputs['image'].ndim - 1

        if keep_aspect_ratio:
            scaling = [_uniform_rand(min_scale, max_scale)] * dim
        else:
            scaling = [_uniform_rand(min_scale, max_scale) for _ in range(dim)]

        target_size = np.array(inputs['image'].shape[:-1]) * scaling
        target_size = np.round(target_size).astype(int).tolist()

        return resize(inputs, target_size=target_size)


def crop_to_roi(
    inputs: dict,
    *,
    roi: Sequence[int] | ArrayLike,
    area_ratio_threshold: float = 1.0,
    clip_bboxes: bool = False,
    p: float = 1.0,
) -> dict:
    """Crop image to bounding-box ROI

    Args:
        inputs: dict

            * image
            * image_mask
            * centroids
            * bboxes
            * cell_masks

    Keyword Args:
        roi: Rectangle roi in yxyx (2d) or zyxzyx (3d) format
        area_ratio_threshold: remove instances if the bbox's relative remaining area is below this threshold
        clip_bboxes: for partially cropped cells, whether to clip bbox and cell_masks to be within the cropped region
        p: probability of applying transformation

    """
    roi = np.array(roi).astype(int)

    def _crop_to_roi(inputs):
        dim = inputs['image'].ndim - 1
        assert dim == 2 or dim == 3, f"invalid image shape {inputs['image'].sahpe}"
        span = roi[dim:] - roi[:dim]
        if (roi < 0).any() or (roi > inputs['image'].shape[:-1] * 2).any() or (span < 0).any():
            raise ValueError(f"invalid roi specified: {list(roi)}")

        if dim == 2:  
            inputs["image"] = inputs["image"][roi[0]:roi[2], roi[1]:roi[3], :]
        else:
            inputs["image"] = inputs["image"][roi[0]:roi[3], roi[1]:roi[4], roi[2]:roi[5],:]

        if "image_mask" in inputs:
            if dim == 2:
                inputs["image_mask"] = inputs["image_mask"][roi[0]:roi[2], roi[1]:roi[3]]
            else:
                inputs["image_mask"] = inputs["image_mask"][roi[0]:roi[3], roi[1]:roi[4], roi[2]:roi[5]]

        boolean_mask = _init_boolean_mask(inputs)

        if boolean_mask is None:
            return inputs

        if "bboxes" in inputs:
            bboxes = inputs["bboxes"] - np.r_[roi[:dim], roi[:dim]]
            areas = _box_area(bboxes)

            bboxes_clipped = np.clip(bboxes, 0, np.r_[span, span])
            areas_clipped = _box_area(bboxes_clipped)

            boolean_mask &= (areas_clipped / areas) >= area_ratio_threshold

            orig_bboxes = inputs["bboxes"]
            if clip_bboxes:
                inputs["bboxes"] = bboxes_clipped[boolean_mask]
            else:
                inputs["bboxes"] = bboxes[boolean_mask]

            if "centroids" in inputs:
                # for clipped cells, use center of bbox as new location
                ctrds = np.where(
                    (areas_clipped < areas)[:, None],
                    bboxes_clipped.reshape(-1, 2, dim).mean(axis=1),
                    inputs["centroids"] - roi[:dim],
                )
                inputs["centroids"] = ctrds[boolean_mask]
                        
            if "cell_masks" in inputs:
                inputs["cell_masks"] = inputs["cell_masks"][boolean_mask]
                if clip_bboxes:
                    target_shape = inputs["cell_masks"].shape[1:]

                    orig_bboxes = orig_bboxes[boolean_mask]
                    orig_r0 = orig_bboxes[:, :dim] - roi[:dim]
                    orig_size = orig_bboxes[:, dim:] - orig_bboxes[:, :dim]

                    new_bbox = inputs["bboxes"] - np.c_[orig_r0, orig_r0]
                    new_bbox = new_bbox / np.c_[orig_size, orig_size]
                    new_bbox = new_bbox * np.r_[target_shape, target_shape]

                    inputs["cell_masks"] = _crop_and_resize(
                        inputs["cell_masks"],
                        new_bbox,
                        target_shape,
                    )

        elif "centroids" in inputs:
            ctrds = inputs["centroids"] - roi[:dim]
            boolean_mask &= (ctrds >= 0).all(axis=-1) & (ctrds < span).all(axis=-1)

            inputs["centroids"] = ctrds[boolean_mask]


        return inputs

    if p < 1.0 and random() >= p:
        return inputs

    else:
        return _crop_to_roi(inputs.copy())


def random_crop(
    inputs: dict,
    *,
    target_size: Sequence[int] | ArrayLike,
    area_ratio_threshold: float = 1.0,
    clip_bboxes: bool = False,
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
        clip_bboxes: for partially cropped cells, whether to clip bbox and cell_masks to be within the cropped region
        p: probability of applying transformation
    """
    dim = inputs['image'].ndim - 1

    if len(target_size) != dim:
        raise ValueError(f"invalid target_size {target_size}")

    target_size = np.array(target_size).astype(int)
    if (target_size <= 0).any():
        raise ValueError(f"invalid target size {list(target_size)}")
    if (target_size > inputs['image'].shape[:-1]).any():
        raise ValueError(f"crop size {target_size} bigger than input size {inputs['image'].shape}")

    if p < 1.0 and random() >= p:

        return inputs

    else:
        r0 = np.array([
            int(random() * (inputs['image'].shape[k] - target_size[k])) 
            for k in range(dim)
        ])
        roi = np.r_[r0, r0 + target_size] 

        return crop_to_roi(
            inputs, roi = tuple(roi),
            area_ratio_threshold=area_ratio_threshold,
            clip_bboxes=clip_bboxes,
        )


def pad(
    inputs: dict,
    *,
    paddings: int|Sequence[int]|ArrayLike,
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
        paddings: either a tuple or a single value. If latter, use the same padding for all axes
        constant_values: the value to fill the padded area
        p: probability of applying transformation
    """
    def _format_padding(p):
        try:
            pl, pr = p
        except:
            pl = pr = int(p) 
        return [pl, pr]

    if p < 1.0 and random() >= p:
        return inputs

    inputs = inputs.copy()
    dim = inputs['image'].ndim - 1
    try:
        iter(paddings)
    except:
        paddings = [int(paddings)] * dim

    paddings = [_format_padding(p) for p in paddings]

    inputs["image"] = np.pad(
        inputs["image"], list(paddings) + [[0, 0]],
        constant_values=constant_values,
    )

    if "image_mask" in inputs:
        inputs["image_mask"] = np.pad(
            inputs["image_mask"], paddings,
            constant_values=constant_values,
        )

    padding_left = [p[0] for p in paddings]
    if "centroids" in inputs:
        inputs["centroids"] = inputs["centroids"] + padding_left

    if "bboxes" in inputs:
        inputs["bboxes"] = inputs["bboxes"] + np.r_[padding_left, padding_left]
                                    
    return inputs


def pad_to_size(
    inputs: dict,
    *,
    target_size: Sequence[int] | ArrayLike,
    constant_values: float = 0,
    padding_type: str = "both",
    p: float = 1.0,
) -> dict:
    """Pad image and labels to a target size.

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes

    Keyword Args:
        target_size: target image size
        constant_values: the value to fill the padded area
        padding_type: "left", "right" or "both"
        p: probability of applying transformation
    """
    if not padding_type in ("left", "right", "both"):
        raise ValueError(f"unknown padding_type {padding_type}")

    if p < 1.0 and random() >= p:
        return inputs

    image = inputs['image']
    paddings = np.array(target_size).astype(int) - np.array(image.shape[:-1])
    if not (paddings >= 0).all():
        raise ValueError(f"target_size {target_size} smaller than input image {image.shape}")
    
    if padding_type == "left":
        paddings = np.c_[paddings, np.zeros_like(paddings)]
    elif padding_type == "right":
        paddings = np.c_[np.zeros_like(paddings), paddings]
    elif padding_type == "both":
        paddings_half = paddings // 2
        paddings = np.c_[paddings_half, paddings - paddings_half]
    else:
        raise ValueError(f"unknown padding type {padding_type}")

    return pad(inputs, paddings=paddings, constant_values=constant_values)


def random_crop_or_pad(
    inputs: dict,
    *,
    target_size: Sequence[int] | ArrayLike,
    constant_values: float = 0,
    padding_type: str = "both",
    area_ratio_threshold: float = 1.0,
    clip_bboxes: bool = False,
    p: float = 1.0,
) -> dict:
    """Random crop or pad image to specified target_size.

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes
            * cell_masks

    Keyword Args:
        target_size: target size
        constant_values: the value to fill the padded area
        padding_type: "left", "right" or "both"
        area_ratio_threshold: remove instances if the bbox's relative remaining area is below this threshold
        clip_bboxes: for partially cropped cells, whether to clip bbox and cell_masks to be within the cropped region
        p: probability of applying transformation
    """
    target_size = np.array(target_size).astype(int)
    if (target_size < 0).any():
        raise ValueError(f"target size cannot be negtive, got {target_size}")

    if p < 1.0 and random() >= p:
        return inputs

    else:
        inputs = pad_to_size(
            inputs, 
            target_size=np.maximum(inputs['image'].shape[:-1], target_size), 
            constant_values=constant_values,
            padding_type=padding_type,
        )

        inputs = random_crop(
            inputs, 
            target_size=target_size, 
            area_ratio_threshold=area_ratio_threshold,
            clip_bboxes=clip_bboxes,
        )

        return inputs



def sample_patch(
    inputs:dict, *, 
    target_size: Sequence[int] | ArrayLike,
    scaling: Sequence[float] | ArrayLike, 
    keep_aspect_ratio: bool = False, 
    constant_values: float = 0,
    padding_type: str = "both",
    area_ratio_threshold: float = 1.0,
    clip_bboxes: bool = False,
    p: float = 1.0
):

    if p < 1.0 and random() >= p:
        return inputs

    min_scale, max_scale = scaling
    dim = inputs['image'].ndim - 1

    assert len(target_size) == dim, f"target_size should have {dim} elements"

    if keep_aspect_ratio:
        scaling = [_uniform_rand(min_scale, max_scale)] * dim
    else:
        scaling = [_uniform_rand(min_scale, max_scale) for _ in range(dim)]

    patch_size = np.round(np.array(target_size) / scaling).astype(int)

    data = random_crop_or_pad(
        inputs, 
        target_size=patch_size.tolist(), 
        constant_values=constant_values, 
        padding_type=padding_type,
        area_ratio_threshold=area_ratio_threshold,
        clip_bboxes=clip_bboxes,
    )

    data = resize(data, target_size=target_size)

    return data   


def flip_left_right(inputs: dict, *, reformat_bboxes:bool=False, p: float = 1.0) -> dict:
    """Flip image left-right

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes
            * cell_masks

    Keyword Args:
        reformat_bboxes: whether to reformat bboxes so that x0 < x1, y0 < y1 etc.
        p: probability of applying transformation

    """
    if p < 1.0 and random() >= p:
        return inputs

    inputs = inputs.copy()
    img_shape = inputs['image'].shape[:-1]
    dim = len(img_shape)
    W = img_shape[-1]

    inputs["image"] = inputs["image"][..., ::-1, :]

    if "image_mask" in inputs:
        inputs["image_mask"] = inputs["image_mask"][..., ::-1]

    if "centroids" in inputs:
        inputs["centroids"][:, -1] = W - inputs["centroids"][:, -1]

    if "bboxes" in inputs:
        if reformat_bboxes:
            inputs["bboxes"][:, [-dim-1, -1]] = W - inputs["bboxes"][:, [-1, -dim-1]]
        else:
            inputs["bboxes"][:, [-dim-1, -1]] = W - inputs["bboxes"][:, [-dim-1, -1]]

    if "cell_masks" in inputs and reformat_bboxes:
        inputs["cell_masks"] = inputs["cell_masks"][..., ::-1]

    return inputs

def flip_up_down(inputs: dict, *, reformat_bboxes:bool=False, p: float = 1.0) -> dict:
    """Flip image up-down

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes
            * cell_masks

    Keyword Args:
        reformat_bboxes: whether to reformat bboxes so that x0 < x1, y0 < y1 etc.
        p: probability of applying transformation

    """
    if p < 1.0 and random() >= p:
        return inputs

    inputs = inputs.copy()
    img_shape = inputs['image'].shape[:-1]
    dim = len(img_shape)
    H = img_shape[-2]

    inputs["image"] = inputs["image"][..., ::-1, :, :]

    if "image_mask" in inputs:
        inputs["image_mask"] = inputs["image_mask"][..., ::-1, :]

    if "centroids" in inputs:
        inputs["centroids"][:, -2] = H - inputs["centroids"][:, -2]

    if "bboxes" in inputs: 
        if reformat_bboxes:
            inputs["bboxes"][:, [-2, -dim-2]] = H - inputs["bboxes"][:, [-dim-2, -2]]
        else:
            inputs["bboxes"][:, [-2, -dim-2]] = H - inputs["bboxes"][:, [-2, -dim-2]]

    if "cell_masks" in inputs and reformat_bboxes:
        inputs["cell_masks"] = inputs["cell_masks"][..., ::-1, :]

    return inputs


def flip_top_bottom(inputs: dict, *, reformat_bboxes:bool=False, p: float = 1.0) -> dict:
    """Flip image top and bottom

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes
            * cell_masks

    Keyword Args:
        reformat_bboxes: whether to reformat bboxes so that x0 < x1, y0 < y1 etc.
        p: probability of applying transformation

    """
    if p < 1.0 and random() >= p:
        return inputs

    inputs = inputs.copy()
    img_shape = inputs['image'].shape[:-1]
    dim = len(img_shape)
    if dim != 3:
        raise ValueError(f"called flip_top_bottom on 2d input")

    D = img_shape[-3]

    inputs["image"] = inputs["image"][::-1, :, :, :]

    if "image_mask" in inputs:
        inputs["image_mask"] = inputs["image_mask"][::-1, :, :]

    if "centroids" in inputs:
        inputs["centroids"][:, -3] = D - inputs["centroids"][:, -3]

    if "bboxes" in inputs: 
        if reformat_bboxes:
            inputs["bboxes"][:, [-3, -dim-3]] = D - inputs["bboxes"][:, [-dim-3, -3]]
        else:
            inputs["bboxes"][:, [-3, -dim-3]] = D - inputs["bboxes"][:, [-3, -dim-3]]

    if "cell_masks" in inputs and reformat_bboxes:
        inputs["cell_masks"] = inputs["cell_masks"][::-1, :, :]

    return inputs


def random_flip(inputs: dict, *, reformat_bboxes:bool=False, p: float = 1.0) -> dict:
    """Flip image on a random axis

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes
            * cell_masks

    Keyword Args:
        reformat_bboxes: whether to reformat bboxes so that x0 < x1, y0 < y1 etc.
        p: probability of applying transformation

    """
    if p < 1.0 and random() >= p:
        return inputs

    img_shape = inputs['image'].shape[:-1]
    dim = len(img_shape)

    flip_axis = int(random() * dim)
    if flip_axis == 0:
        return flip_left_right(inputs, reformat_bboxes=reformat_bboxes)
    elif flip_axis == 1:
        return flip_up_down(inputs, reformat_bboxes=reformat_bboxes)
    else:
        return flip_top_bottom(inputs, reformat_bboxes=reformat_bboxes)


def swapaxes(inputs: dict, *, axes:str="xy", p:float=1):
    """tranpose image on a random axis

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes
            * cell_masks

    Keyword Args:
        axes: "xy", "yz" or "xz". must be "xy" for 2d 
        p: probability of applying transformation
    """
    if p < 1.0 and random() >= p:
        return inputs
    
    inputs = inputs.copy()
    img_shape = inputs['image'].shape[:-1]
    dim = len(img_shape)

    assert dim ==2 or dim == 3, f"unexpected image dim {img_shape}"

    if (dim == 2 and axes != "xy") or not axes in ("xy", "xz", "yz"):
        raise ValueError(f"invalid axes {axes}")

    lut = {"x":2, "y":1, "z":0} if dim == 3 else {"x":1, "y":0}
    a1, a2 = (lut[x] for x in axes)
    
    inputs["image"] = np.swapaxes(inputs["image"], a1, a2)

    if "image_mask" in inputs:
        inputs["image_mask"] = np.swapaxes(inputs["image_mask"], a1, a2)

    if "centroids" in inputs:
        inputs["centroids"] = inputs["centroids"].copy()
        inputs["centroids"][:, [a1, a2]] = inputs["centroids"][:, [a2, a1]]

    if "bboxes" in inputs: 
        inputs["bboxes"] = inputs['bboxes'].copy()
        inputs["bboxes"][:, [a1, a2, a1+dim, a2+dim]] = inputs["bboxes"][:, [a2, a1, a2+dim, a1+dim]]

    if "cell_masks" in inputs:
        inputs["cell_masks"] = np.swapaxes(inputs["cell_masks"], a1+1, a2+1)

    return inputs

def random_swapaxes(inputs: dict, *, p:float=1):
    """tranpose image on a random axis

    Args:
        inputs: dict data:

            * image
            * image_mask
            * centroids
            * bboxes
            * cell_masks

    Keyword Args:
        p: probability of applying transformation
    """
    if p < 1.0 and random() >= p:
        return inputs

    img_shape = inputs['image'].shape[:-1]
    dim = len(img_shape)

    if dim == 2:
        return swapaxes(inputs)

    choices = ("xy", "yz", "xz")
    axes = choices[int(random() * 3)]
    return swapaxes(inputs, axes=axes)


def random_cut_out(inputs:dict, *, size:int, n:int = 1, p:float=1):
    if p < 1.0 and random() >= p:
        return inputs

    image = inputs['image'].copy()
    img_shape = image.shape[:-1]
    size = np.array(size).astype(int)

    sel = np.zeros([inputs['bboxes'].shape[0]], dtype=int)
    sel[:n] = 1
    np.random.shuffle(sel)
    sel_boxes = inputs['bboxes'][sel > 0]
    dim = sel_boxes.shape[-1]//2
    mins = np.minimum(sel_boxes[:,:dim], sel_boxes[:,dim:])
    maxs = np.maximum(sel_boxes[:,:dim], sel_boxes[:,dim:])
    c0s = np.random.randint(mins-size, maxs+size)

    for c0 in c0s:
        s = tuple(slice(max(0, a), min(c, b)) for a, b, c in zip(c0, c0+size, img_shape))
        image.__setitem__(s, 0)

    inputs['image'] = image

    return inputs
    

def mosaic(inputs: Sequence[dict]|dict):
    import jax
    if not isinstance(inputs, dict):
        assert len(inputs) == 4, f"mosaic requires a sequence of length 4, got{len(inputs)}"
        inputs = jax.tree_util.tree_map(lambda *x: x, *inputs) # ensure same pytree

    # tile
    images = inputs['image']
    img_shape = images[0].shape[:-1]
    image_tile = np.concatenate([
        np.concatenate(images[0:2], axis=-2),
        np.concatenate(images[2:4], axis=-2),
    ], axis=-3)
    del inputs['image']

    if "image_mask" in inputs:
        images = inputs['image_mask']
        image_mask_tile = np.concatenate([
            np.concatenate(images[0:2], axis=-1),
            np.concatenate(images[2:4], axis=-1),
        ], axis=-2)
        del inputs['image_mask']

    H, W = img_shape[-2:]
    offsets = np.array([
        [0, 0],
        [0, W],
        [H, 0],
        [H, W],
    ])
    if len(img_shape) == 3:
        offsets = np.c_[np.zeros([4, 1], dtype=int), offsets]
    
    if "centroids" in inputs:
        inputs["centroids"] = [
            locs + ofs
            for locs, ofs in zip(inputs["centroids"], offsets)
        ]
        
    if "bboxes" in inputs:
        inputs["bboxes"] = [
            box + np.r_[ofs, ofs]
            for box, ofs in zip(inputs["bboxes"], offsets)
        ]

    outputs = {k: np.concatenate(v) for k, v in inputs.items()}
    outputs['image'] = image_tile
    if "image_mask" in inputs:
        outputs["image_mask"] = image_mask_tile

    return outputs
