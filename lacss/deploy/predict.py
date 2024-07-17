""" 
Attributes:
    model_urls: URLs for build-in pretrain models. e.g model_urls["default"].
"""
from __future__ import annotations

import logging
import math
import pickle
from functools import lru_cache, partial, reduce
from typing import Mapping, Optional, Sequence, Tuple, Union

import cv2
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import freeze, unfreeze

import lacss.modules
import lacss.ops

from ..ops import patches_to_label, non_max_suppression
from ..typing import *
from ..utils import load_from_pretrained

Shape = Sequence[int]

_cached_partial = lru_cache(partial)

# model_urls: Mapping[str, str] = {
#     "cnsp4-bf": "https://huggingface.co/jiyuuchc/lacss-cnsp4-bf/resolve/main/cnsp4_bf.bin?download=true",
#     "cnsp4-fl": "https://huggingface.co/jiyuuchc/lacss-cnsp4-fl/resolve/main/cnsp4_fl.bin?download=true",
#     "cnsp4-base": "https://huggingface.co/jiyuuchc/lacss-cnsp4-base/resolve/main/cnsp4_base.bin?download=true",
# }
model_urls: Mapping[str, str] = {
    "lacss2s-n-bf": "https://huggingface.co/jiyuuchc/lacss2s-n-bf/resolve/main/lacss2s-n-bf?download=true",
}
# model_urls["default"] = model_urls["cnsp4-base"]


def _to_polygons(
    pred,
    *,
    segmentation_threshold=0.5,
    scaling=1.0,
    chain_approx=cv2.CHAIN_APPROX_SIMPLE,
) -> list:
    polygons = []

    y0s = np.asarray(pred["y0s"])
    x0s = np.asarray(pred["x0s"])
    segs = np.asarray(pred["segmentations"] >= segmentation_threshold).astype("uint8")

    if "instance_mask" in pred:
        mask = np.asarray(pred["instance_mask"])
        segs = segs[mask]
        y0s = y0s[mask]
        x0s = x0s[mask]

    for y0, x0, seg in zip(y0s, x0s, segs):
        c, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, chain_approx)

        max_len_element = reduce(
            lambda a, b: a if len(a) >= len(b) else b,
            c,
            np.zeros([0, 1, 2], dtype=int),
        )

        polygon = max_len_element.squeeze(1).astype(float) + [x0, y0]

        polygons.append(polygon)

    if scaling != 1.0:
        polygons = [(c + 0.5) / scaling - 0.5 for c in polygons]

    return polygons


def _draw_label(output, segmentation_threshold, scaling, image_size) -> np.ndarray:
    polygons = _to_polygons(
        output,
        segmentation_threshold=segmentation_threshold,
        scaling=scaling,
    )
    label = np.zeros(image_size, dtype="int32")
    color = len(polygons)
    for polygon in polygons[::-1]:
        if len(polygon) > 0:
            polygon = np.round(polygon).astype(int)
            cv2.fillPoly(label, [polygon], color)
        color -= 1

    return label


def _remove_edge_instances(
    pred: dict,
    img_shape: Tuple[int, int],
    remove_top: bool = True,
    remove_bottom: bool = True,
    remove_left: bool = True,
    remove_right: bool = True,
    threshold: float = 0.5,
    min_border_pixel: int = 1,
):
    h, w = img_shape
    segs = pred["segmentations"] >= threshold
    yc, xc = jnp.mgrid[: segs.shape[-2], : segs.shape[-1]]
    yc += pred["y0s"][:, None, None]
    xc += pred["x0s"][:, None, None]

    removal = jnp.zeros([len(segs)], dtype=bool)

    removal |= remove_top & (
        jnp.count_nonzero((yc <= 0) & segs, axis=(1, 2)) >= min_border_pixel
    )

    removal |= remove_bottom & (
        jnp.count_nonzero((yc >= h - 1) & segs, axis=(1, 2)) >= min_border_pixel
    )

    removal |= remove_left & (
        jnp.count_nonzero((xc <= 0) & segs, axis=(1, 2)) >= min_border_pixel
    )

    removal |= remove_right & (
        jnp.count_nonzero((xc >= w - 1) & segs, axis=(1, 2)) >= min_border_pixel
    )

    removal = removal.reshape(pred["instance_mask"].shape)

    pred["instance_mask"] &= ~removal

    return pred


@partial(jax.jit, static_argnums=0)
def _predict(context, params, image):
    image_in = image
    scaling = context["scaling"]
    apply_fn = context["apply_fn"]
    output_type = context["output_type"]
    nms_iou = context["nms_iou"]

    if scaling != 1:
        image = jax.image.resize(
            image,
            [
                round(image.shape[0] * scaling),
                round(image.shape[1] * scaling),
                image.shape[2],
            ],
            "linear",
        )

    orig_h, orig_w = image.shape[:2]
    if orig_h // 32 * 32 != orig_h or orig_w // 32 * 32 != orig_w:
        padded_h = orig_h // 32 * 32 + 32
        padded_w = orig_w // 32 * 32 + 32
        image = jnp.pad(image, [[0, padded_h - orig_h], [0, padded_w - orig_w], [0, 0]])

    preds = apply_fn(dict(params=params), image)["predictions"]

    instance_mask = preds["segmentation_is_valid"]
    instance_mask &= preds["scores"] >= context["score_threshold"]

    if context["min_area"] > 0:
        areas = jnp.count_nonzero(
            preds["segmentations"] > context["segmentation_threshold"], axis=(1, 2)
        )
        instance_mask &= areas > context["min_area"]

    if context["remove_out_of_bound"]:
        pred_locations = preds["pred_locations"]
        valid_locs = (pred_locations >= 0).all(axis=-1)
        valid_locs &= pred_locations[:, 0] < orig_h
        valid_locs &= pred_locations[:, 1] < orig_w
        instance_mask &= valid_locs

    # note: convert logit to p in output
    output = dict(
        instance_mask=instance_mask,
        segmentations=jax.nn.sigmoid(preds["segmentations"]),
        y0s=preds["segmentation_y0_coord"],
        x0s=preds["segmentation_x0_coord"],
        scores=preds["scores"],
        bboxes=lacss.ops.bboxes_of_patches(
            preds, threshold=context["segmentation_threshold"]
        ),
    )

    if output_type == "bbox":
        output["segmentations"] = jax.vmap(_rescale_masks)(
            output["segmentations"],
            output["y0s"],
            output["x0s"],
            output["bboxes"],
        )

    if nms_iou > 0 and output_type != "grid":  # need nms
        boxes = output["bboxes"]
        mask = output["instance_mask"]
        boxes = jnp.where(mask[:, None], boxes, -1)
        _, _, selections = non_max_suppression(
            output["scores"],
            boxes,
            -1,
            threshold=nms_iou,
            return_selection=True,
        )
        output["instance_mask"] &= selections

    if output_type == "raw":
        if scaling != 1.0:
            seg, yc, xc = lacss.ops.rescale_patches(preds, 1 / scaling)
            output["segmentations"] = seg
            output["y0s"] = yc[:, 0, 0]
            output["x0s"] = xc[:, 0, 0]

    # try to compute label if scaling is not extreme
    # otherwise compute label using polygon later to avoid memory overflow
    if output_type == "label" and scaling < 2.0 and scaling > 0.5:
        st = context["segmentation_threshold"]
        if scaling != 1.0:
            (
                preds["segmentations"],
                preds["segmentation_y_coords"],
                preds["segmentation_y_coords"],
            ) = lacss.ops.rescale_patches(preds, 1 / scaling)
            # rescale_patches convert logits to p
            st = jax.nn.sigmoid(st)

        label = lacss.ops.patches_to_label(
            preds,
            input_size=image_in.shape[:2],
            mask=output["instance_mask"],
            score_threshold=0,
            threshold=st,
        )
        output.update(dict(label=label.astype(int)))

    return output


def _rescale_masks(seg, y0, x0, bbox, mask_dim=[48, 48]):
    by0 = bbox[0] - y0
    bx0 = bbox[1] - x0
    bh = bbox[2] - bbox[0]
    bw = bbox[3] - bbox[1]

    mg = jnp.mgrid[: mask_dim[0], : mask_dim[1]].transpose((1, 2, 0)) + 0.5
    mg /= jnp.asarray(mask_dim)

    mg = mg * jnp.asarray([bh, bw]) + jnp.asarray([by0, bx0])

    return lacss.ops.sub_pixel_samples(seg, mg, edge_indexing=True)


class Predictor:
    """Main class interface for model deployment. This is the only class you
    need if you don't train your own model

    Examples:
        The most common use case is to use a build-in pretrained model.

            import lacss.deploy

            # look up the url of a build-in mode
            url = lacss.deploy.model_urls["default"]

            # create the predictor instance
            predictor = lacss.deploy.Predictor(url)

            # make a prediction
            label = predictor.predict(image)

    Attributes:
        module: The underlying FLAX module
        params: Model weights.
        detector: The detector submodule for convinence. A common use case
            is to customize this submodule during inference. e.g.
            ```
            predictor.detector.test_min_score = 0.5
            ```
            Detector submodule has no trained paramters.
    """

    def __init__(
            self, 
            url: str | tuple[nn.Module, dict],
            *,
            precompile_shape: Optional[Shape] = None
        ):
        """Construct Predictor

        Args:
            url: A URL or local path to the saved model.
                URLs for build-in pretrained models can be found in lacss.deploy.model_urls

        Keyword Args:
            precompile_shape: Image shape(s) for precompiling. Otherwise
                the model will be recompiled for every new input image shape.

        """

        if isinstance(url, tuple) and len(url) == 2:
            self.model = url
            if not isinstance(self.model[0], nn.Module):
                raise ValueError(
                    "Initiaize the Predictor with a tuple, but the first element is not a Module."
                )
        else:
            self.model = load_from_pretrained(url)

        if precompile_shape is not None:
            logging.info("Prcompile the predictor for image shape {precompile_shape}.")
            logging.info("This will take several minutes.")

            try:
                iter(precompile_shape[0])
            except:
                precompile_shape = [precompile_shape]
            for shape in precompile_shape:
                x = jnp.zeros(shape)
                _ = self.predict(x)

    def predict(
        self,
        image: ArrayLike,
        *,
        min_area: float = 0,
        remove_out_of_bound: bool = False,
        scaling: float = 1.0,
        output_type: str = "label",
        score_threshold: float = 0.5,
        segmentation_threshold: float = 0.5,
        nms_iou: float = 1,
        **kwargs,
    ) -> dict:
        """Predict segmentation.

        Args:
            image: A ndarray of (h,w,c) format. Value of c must be 1-3

        Keyword Args:
            output_type: "label" | "contour" | "bbox"
            min_area: Minimum area of a valid prediction.
            scaling: A image scaling factor. If not 1, the input image will be resized internally before fed
                to the model. The results will be resized back to the scale of the orginal input image.
            score_threshold: Min score needed to be included in the output.
            segmentation_threshold: Threshold value for segmentation
            nms_iou: IOU threshold value for non-max-supression durint post-processing

        Returns:
            For "label" output:

                - pred_scores: The prediction scores of each instance.
                - pred_label: a 2D image label. 0 is background.

            For "contour" output:

                - pred_scores: The prediction scores of each instance.
                - pred_contours: a list of polygon arrays in x-y format.

            For "bbox" output (ie MaskRCNN):

                - pred_scores: The prediction scores of each instance.
                - pred_bboxes: The bounding-boxes of detected instances in y0x0y1x1 format
                - pred_masks:  A 3d array representing (rescaled) segmentation mask within bboxes
        """
        if not output_type in ("bbox", "label", "contour", "raw"):
            raise ValueError(
                f"output_type should be 'bbox'|'label'|'contour'. Got {output_type} instead."
            )

        module, params = self.model

        if len(kwargs) > 0:
            apply_fn = _cached_partial(module.apply, **kwargs)
        else:
            apply_fn = module.apply

        inv_sig_st = math.log(segmentation_threshold / (1 - segmentation_threshold))
        ctx = freeze(
            dict(
                apply_fn=apply_fn,
                output_type=output_type,
                segmentation_threshold=inv_sig_st,
                score_threshold=score_threshold,
                scaling=scaling,
                min_area=min_area,
                remove_out_of_bound=remove_out_of_bound,
                nms_iou=nms_iou,
            )
        )
        output = _predict(ctx, params, image)
        mask = np.asarray(output["instance_mask"])
        scores = np.asarray(output["scores"])[mask]
        bboxes = np.asarray(output["bboxes"])[mask] / scaling

        if output_type == "bbox":
            return dict(
                pred_scores=scores,
                pred_bboxes=bboxes,
                pred_masks=np.asarray(output["segmentations"])[mask],
            )

        elif output_type == "contour":
            contours = _to_polygons(
                output,
                segmentation_threshold=segmentation_threshold,
                scaling=scaling,
            )

            return dict(
                pred_scores=scores,
                pred_bboxe=bboxes,
                pred_contours=contours,
            )
        elif output_type == "raw":
            return dict(
                pred_scores=scores,
                pred_bboxes=bboxes,
                pred_masks=np.asarray(output["segmentations"])[mask],
                y0s=np.asarray(output["y0s"])[mask],
                x0s=np.asarray(output["x0s"])[mask],
            )
        else:  # Label
            if "label" in output:
                label = np.asarray(output["label"])
            else:  # compute via polygons
                label = _draw_label(
                    output,
                    segmentation_threshold,
                    scaling,
                    image.shape[:2],
                )

            return dict(
                pred_scores=scores,
                pred_label=label,
            )

    def predict_on_large_image(
        self,
        image: ArrayLike,
        gs: int,
        ss: int,
        *,
        scaling: float = 1,
        min_area: int = 0,
        nms_iou: float = 0.25,
        segmentation_threshold: float = 0.5,
        score_threshold: float = 0.5,
        output_type: str = "label",
        min_cells_per_patch: int = 0,
        disable_padding: bool = False,
        **kwargs,
    ) -> dict:
        """Make prediction on very large image by dividing into a grid.

        Direct model prediction on large image may cause out-of-memory error. This
        method divided the large image to smaller grid and then stitch the results
        to form the complete prediction.

        Args:
            image: An image with (H, W, C) format.
            gs: An int value. Grid size of the computation.
            ss: An int value of stepping size. Must be small than gs to produce valid
                results.

        Keyword Args:
            output_type: label" | "contour" | "bbox"
            min_area: Minimum area of a valid prediction.
            scaling: A image scaling factor. If not 1, the input image will be resized internally before fed
                to the model. The results will be resized back to the scale of the orginal input image.
            score_threshold: Min score needed to be included in the output.
            segmentation_threshold: Threshold value for segmentation.
            nms_iou: IOU threshold value for non-max-supression durint post-processing

        Returns:
            For "label" output:

                - pred_scores: The prediction scores of each instance.
                - pred_label: a 2D image label. 0 is background.

            For "contour" output:

                - pred_scores: The prediction scores of each instance.
                - pred_contours: a list of polygon arrays in x-y format.

            For "bbox" output (ie MaskRCNN):

                - pred_scores: The prediction scores of each instance.
                - pred_bboxes: The bounding-boxes of detected instances in y0x0y1x1 format
                - pred_masks: A 3d array representing segmentation within bboxes
        """

        if not output_type in ("bbox", "label", "contour"):
            raise ValueError(
                f"output_type should be 'patch'|'label'|'contour'. Got {output_type} instead."
            )

        module, params = self.model

        if len(kwargs) > 0:
            apply_fn = _cached_partial(module.apply, **kwargs)
        else:
            apply_fn = module.apply

        inv_sig_st = math.log(segmentation_threshold / (1 - segmentation_threshold))
        ctx = freeze(
            dict(
                apply_fn=apply_fn,
                output_type="bbox" if output_type == "bbox" else "grid",
                segmentation_threshold=inv_sig_st,
                score_threshold=score_threshold,
                scaling=scaling,
                min_area=min_area,
                remove_out_of_bound=True,
                nms_iou=0,
            )
        )

        h, w = image.shape[:2]

        if not disable_padding:
            get_padding = lambda a, gs, ss: (a - 1) // ss * ss + gs - a

            padded_img = np.pad(
                image,
                [[0, get_padding(h, gs, ss)], [0, get_padding(w, gs, ss)], [0, 0]],
            )
        else:
            padded_img = image

        preds = []
        for y0 in range(0, h, ss):
            for x0 in range(0, w, ss):
                logging.info(f"Processing grid {y0}-{x0}")
                pred = _predict(
                    ctx,
                    params,
                    padded_img[y0 : y0 + gs, x0 : x0 + gs],
                )
                pred = _remove_edge_instances(
                    pred,
                    [gs, gs],
                    remove_top=y0 > 0,
                    remove_bottom=(y0 + gs) < h,
                    remove_left=x0 > 0,
                    remove_right=(x0 + gs) < w,
                    threshold=segmentation_threshold,
                )

                logging.info(f"Transfer result...")

                mask = np.asarray(pred["instance_mask"])

                if np.count_nonzero(mask) > min_cells_per_patch:
                    if output_type != "bbox":
                        pred = dict(
                            scores=np.array(pred["scores"])[mask],
                            bboxes=np.array(pred["bboxes"])[mask] + [y0, x0, y0, x0],
                            segmentations=np.array(pred["segmentations"])[mask],
                            y0s=np.array(pred["y0s"])[mask] + y0,
                            x0s=np.array(pred["x0s"])[mask] + x0,
                        )

                        polygons = _to_polygons(
                            pred,
                            segmentation_threshold=segmentation_threshold,
                            scaling=scaling,
                        )

                        valid_polygons = np.asarray([len(p) > 0 for p in polygons])
                        scores = np.where(valid_polygons, pred["scores"], -1)
                        bboxes = np.where(
                            valid_polygons[:, None], pred["bboxes"] / scaling, -1
                        )

                        preds.append(
                            dict(
                                scores=scores,
                                polygons=np.fromiter(polygons, dtype=object),
                                bboxes=bboxes,
                            )
                        )

                    else:
                        bboxes = np.array(pred["bboxes"])[mask] + [y0, x0, y0, x0]
                        bboxes /= scaling
                        preds.append(
                            dict(
                                scores=np.array(pred["scores"])[mask],
                                bboxes=bboxes,
                                segmentations=np.array(pred["segmentations"])[mask],
                            )
                        )

        logging.info(f"Postprocessing...")
        preds = jax.tree_util.tree_map(lambda *x: np.concatenate(x), *preds)

        # sort based on scores
        asort = np.argsort(preds["scores"])[::-1]
        preds = jax.tree_util.tree_map(lambda x: x[asort], preds)

        # nms
        if nms_iou > 0:
            logging.info(f"nms...")
            scores = preds["scores"]
            boxes = preds["bboxes"]
            _, _, selections = non_max_suppression(
                scores,
                boxes,
                -1,
                threshold=nms_iou,
                min_score=0,
                return_selection=True,
            )

            preds = jax.tree_util.tree_map(lambda x: x[selections], preds)

        if output_type == "bbox":  # FIXME unimplemented

            return dict(
                pred_scores=preds["scores"],
                pred_bboxes=preds["bboxes"],
                pred_masks=segs,
            )

        if output_type == "label":
            logging.info(f"Generating label...")

            label = _draw_label(preds["polygons"], image.shape[:2])

            return dict(
                pred_scores=preds["scores"],
                pred_label=label,
            )

        else:  # contours
            return dict(
                pred_scores=preds["scores"],
                pred_contours=preds["polygons"],
            )

    @property
    def module(self) -> nn.Module:
        return self.model[0]

    @property
    def params(self) -> Params:
        return self.model[1]

    @params.setter
    def params(self, new_params):
        self.model = self.module, new_params

    @property
    def detector(self) -> lacss.modules.Detector:
        return self.module.detector


    def save(self, save_path) -> None:
        """Re-save the model by pickling.

        In the form of (module, weights).

        Args:
            save_path: Path to the pkl file
        """
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
