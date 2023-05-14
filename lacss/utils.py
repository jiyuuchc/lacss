import dataclasses
import re
import typing as tp

import jax.numpy as jnp
import numpy as np
from flax import struct

from .ops import bboxes_of_patches

InputLike = tp.Union[tp.Any, tp.Tuple[tp.Any, ...], tp.Dict[str, tp.Any], "Inputs"]


def _unique_name(
    names: tp.Set[str],
    name: str,
):

    if name in names:

        match = re.match(r"(.*?)(\d*)$", name)
        assert match is not None

        name = match[1]
        num_part = match[2]

        i = int(num_part) if num_part else 2
        str_template = f"{{name}}{{i:0{len(num_part)}}}"

        while str_template.format(name=name, i=i) in names:
            i += 1

        name = str_template.format(name=name, i=i)

    names.add(name)
    return name


def _unique_names(
    names: tp.Iterable[str],
    *,
    existing_names: tp.Optional[tp.Set[str]] = None,
) -> tp.Iterable[str]:
    if existing_names is None:
        existing_names = set()

    for name in names:
        yield _unique_name(existing_names, name)


def _lower_snake_case(s: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    parts = s.split("_")
    output_parts = []

    for i in range(len(parts)):
        if i == 0 or len(parts[i - 1]) > 1:
            output_parts.append(parts[i])
        else:
            output_parts[-1] += parts[i]

    return "_".join(output_parts)


def _get_name(obj) -> str:
    if hasattr(obj, "name") and obj.name:
        return obj.name
    elif hasattr(obj, "__name__") and obj.__name__:
        return _lower_snake_case(obj.__name__)
    elif hasattr(obj, "__class__") and obj.__class__.__name__:
        return _lower_snake_case(obj.__class__.__name__)
    else:
        raise ValueError(f"Could not get name for: {obj}")


class Inputs(struct.PyTreeNode):
    args: tp.Tuple[tp.Any, ...] = ()
    kwargs: tp.Dict[str, tp.Any] = dataclasses.field(default_factory=dict)

    def update(self, *args, **kwargs):
        tmp = self.kwargs.copy()
        tmp.update(kwargs)
        new_inputs = self.replace(args=self.args + args, kwargs=tmp)
        return new_inputs

    @classmethod
    def from_value(cls, value: InputLike) -> "Inputs":
        if isinstance(value, cls):
            return value
        elif isinstance(value, tuple):
            return cls(args=value)
        elif isinstance(value, dict):
            return cls(kwargs=value)
        else:
            return cls(args=(value,))


def _to_str(p):
    return "".join(p.astype(int).reshape(-1).astype(str).tolist())


def format_predictions(pred, mask=None, encode_patch=True, threshold=0.5):
    """Produce more readable data from model predictions
    Args:
        pred: model output without batch dim
        mask: optional mask selecting cells
        threshold: float patch threshold
    Returns: dict(locations, scores, centroids, bboxes, encodings)
    """

    patches = pred["instance_output"] >= threshold
    yc = pred["instance_yc"]
    xc = pred["instance_xc"]

    bboxes = bboxes_of_patches(pred)

    n_pixels = np.count_nonzero(patches, axis=(1, 2))
    centroids = jnp.stack(
        [
            (patches * yc).sum(axis=(1, 2)) / n_pixels,
            (patches * xc).sum(axis=(1, 2)) / n_pixels,
        ],
        axis=-1,
    )  # centroid

    outputs = dict(
        locations=pred["pred_locations"],
        scores=pred["pred_scores"],
        centroids=centroids,
        bboxes=bboxes,
    )

    is_valid = pred["instance_mask"].squeeze(axis=(-1, -2))
    is_valid &= patches.any(axis=(1, 2))  # no empty patches
    if mask is not None:
        is_valid &= mask

    outputs = {k: np.asarray(v)[is_valid] for k, v in outputs.items()}

    if encode_patch:

        encodings = []
        patches = np.asarray(patches)[is_valid]
        y0 = np.asarray(yc)[is_valid, 0, 0]
        x0 = np.asarray(xc)[is_valid, 0, 0]
        boxes = np.asarray(outputs["bboxes"]) - np.stack([y0, x0, y0, x0], axis=-1)

        for box, patch in zip(boxes, patches):
            roi = patch[box[0] : box[2], box[1] : box[3]]
            encodings.append(_to_str(roi))

        outputs["encodings"] = encodings

    return outputs


def show_images(imgs, locs=None, **kwargs):

    import matplotlib.patches
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 5))
    if len(imgs) == 1:
        axs = [axs]

    for k, img in enumerate(imgs):
        axs[k].imshow(img, **kwargs)
        axs[k].axis("off")
        if locs is not None and locs[k] is not None:
            loc = np.round(locs[k]).astype(int)
            for p in loc:
                c = matplotlib.patches.Circle(
                    (p[1], p[0]), fill=False, edgecolor="white"
                )
                axs[k].add_patch(c)
    plt.tight_layout()
