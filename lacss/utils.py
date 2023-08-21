import dataclasses

import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import freeze, unfreeze


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
    from lacss.ops import bboxes_of_patches

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


def dataclass_from_dict(klass, dikt):
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{f: dataclass_from_dict(fieldtypes[f], dikt[f]) for f in dikt})
    except:
        return dikt


def load_from_pretrained(pretrained: str):
    """Load a saved model.

    Args:
        pretrained: The url to the saved model.

    Returns: A tuple (module, parameters) representing the model.
    """
    import os

    if os.path.isdir(pretrained):
        # directory are orbax checkpoint

        from lacss.train import LacssTrainer

        trainer = LacssTrainer.from_checkpoint(pretrained)
        module = trainer.model.principal
        params = trainer.params["principal"]

    else:
        # uri or files were treated as pickled byte steam
        import pickle

        from .modules import Lacss
        from .train import Trainer

        if os.path.isfile(pretrained):
            with open(pretrained, "rb") as f:
                thingy = pickle.load(f)

        else:
            from urllib.request import Request, urlopen

            headers = {"User-Agent": "Wget/1.13.4 (linux-gnu)"}
            req = Request(url=pretrained, headers=headers)

            bytes = urlopen(req).read()
            thingy = pickle.loads(bytes)

        if isinstance(thingy, Trainer):
            module = thingy.model
            params = thingy.params

        else:
            cfg, params = thingy

            if isinstance(cfg, Lacss):
                module = cfg
            else:
                module = Lacss.from_config(cfg)

    if "params" in params and len(params) == 1:
        params = params["params"]

    return module, freeze(params)


def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    """Packs user-provided data into a tuple."""
    if y is None:
        return (x,)
    elif sample_weight is None:
        return (x, y)
    else:
        return (x, y, sample_weight)


def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple."""
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])

    raise ValueError("Data not understood.")
