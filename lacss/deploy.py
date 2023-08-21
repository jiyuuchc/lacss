""" 
Attributes:
    model_urls: URLs for build-in pretrain models. e.g model_urls["livecell"].
"""
from __future__ import annotations

import logging
import pickle
from functools import lru_cache, partial
from typing import Mapping, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import freeze, unfreeze

import lacss.modules
import lacss.ops

from .ops import patches_to_label, sorted_non_max_suppression
from .typing import *
from .utils import load_from_pretrained

Shape = Sequence[int]

_cached_partial = lru_cache(partial)

model_urls: Mapping[str, str] = {
    "livecell": "https://data.mendeley.com/public-files/datasets/sj3vrvm6w3/files/439e524f-e4e9-4f97-9f38-c22cb85adbd1/file_downloaded",
    "tissuenet": "https://data.mendeley.com/public-files/datasets/sj3vrvm6w3/files/1e0a839d-f564-4ee0-a4f3-34792df7c613/file_downloaded",
}


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
    segs = pred["instance_output"] >= threshold
    yc = pred["instance_yc"]
    xc = pred["instance_xc"]

    removal = jnp.zeros([len(segs)], dtype=bool)
    if remove_top:
        is_top = jnp.count_nonzero((yc <= 0) & segs, axis=(1, 2)) >= min_border_pixel
        removal |= is_top
    if remove_bottom:
        is_bottom = (
            jnp.count_nonzero((yc >= h - 1) & segs, axis=(1, 2)) >= min_border_pixel
        )
        removal |= is_bottom
    if remove_left:
        is_left = jnp.count_nonzero((xc <= 0) & segs, axis=(1, 2)) >= min_border_pixel
        removal |= is_left
    if remove_right:
        is_right = (
            jnp.count_nonzero((xc >= w - 1) & segs, axis=(1, 2)) >= min_border_pixel
        )
        removal |= is_right

    removal = removal.reshape(pred["instance_mask"].shape)
    pred["instance_mask"] &= ~removal

    return pred


@partial(jax.jit, static_argnums=(0, 3))
def _predict(apply_fn, params, image, scaling):
    if scaling != 1:
        h, w, c = image.shape
        image = jax.image.resize(
            image, [round(h * scaling), round(w * scaling), c], "linear"
        )

    preds = apply_fn(dict(params=params), image)
    preds["pred_bboxes"] = lacss.ops.bboxes_of_patches(preds)

    del preds["encoder_features"]
    del preds["decoder_features"]
    del preds["lpn_features"]
    del preds["lpn_scores"]
    del preds["lpn_regressions"]
    del preds["instance_logit"]

    if scaling != 1:
        (
            preds["instance_output"],
            preds["instance_yc"],
            preds["instance_xc"],
        ) = lacss.ops.rescale_patches(preds, 1 / scaling)

        preds["pred_locations"] = preds["pred_locations"] / scaling
        preds["pred_bboxes"] = preds["pred_bboxes"] / scaling

    return preds


class Predictor:
    """Main class interface for model deployment. This is the only class you
    need if you don't train your own model

    Examples:
        The most common use case is to use a build-in pretrained model.

            import lacss.deploy

            # look up the url of a build-in mode
            url = lacss.deploy.model_urls["livecell"]

            # create the predictor instance
            predictor = lacss.deploy.Predictor(url)

            # make a prediction
            label = predictor.predict_label(image)

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

    def __init__(self, url: str, precompile_shape: Optional[Shape] = None):
        """Construct Predictor

        Args:
            url: A URL or local path to the saved model.
                URLs for build-in pretrained models can be found in lacss.deploy.model_urls
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
        return_label: bool = False,
        **kwargs,
    ) -> Union[dict, Array]:
        """Predict segmentation.

        Args:
            image: A ndarray of (h,w,c) format. Value of c must be 1-3

        Keyword Args:
            min_area: Minimum area of a valid prediction.
            remove_out_of_bound: Whether to remove out-of-bound predictions. Default is False.
            scaling: A image scaling factor. If not 1, the input image will be resized internally before fed
                to the model. The results will be resized back to the scale of the orginal input image.
            return_label: Whether to output full model prediction or segmentation label.

        Returns:

            If return_label is False, returns model predictions with following elements:

                - pred_scores: The prediction scores of each instance.
                - pred_bboxes: The bounding-boxes of detected instances in y0x0y1x1 format
                - instance_output: A 3d array representing segmentation instances.
                - instance_yc: The meshgrid y-coordinates of the instances.
                - instance_xc: The meshgrid x-coordinates of the instances.
                - instance_mask: a masking tensor. Invalid instances are maked with 0.

            otherwise return a segmentation label.
        """

        module, params = self.model

        if len(kwargs) > 0:
            apply_fn = _cached_partial(module.apply, **kwargs)
        else:
            apply_fn = module.apply

        preds = _predict(apply_fn, params, image, scaling)

        # check prediction validity
        instance_mask = preds["instance_mask"].squeeze(axis=(1, 2))

        if min_area > 0:
            areas = np.count_nonzero(preds["instance_output"] > 0.5, axis=(1, 2))
            instance_mask &= areas > min_area

        if remove_out_of_bound:
            pred_locations = preds["pred_locations"]
            h, w, _ = image.shape
            valid_locs = (pred_locations >= 0).all(axis=-1)
            valid_locs &= pred_locations[:, 0] < h
            valid_locs &= pred_locations[:, 1] < w
            instance_mask &= valid_locs

        preds["instance_mask"] = instance_mask.reshape(-1, 1, 1)

        if return_label:
            return patches_to_label(
                preds,
                input_size=image.shape[:2],
                score_threshold=0.5,
            )
        else:
            return preds

    def predict_label(
        self, image: ArrayLike, *, score_threshold: float = 0.5, **kwargs
    ) -> Array:
        """Predict segmentation in image label format

        Args:
            image: Input image.
            score_threshold: The minimal prediction scores.

        Returns:
            A [H, W] array. The values indicate the id of the cells. For cells
                with overlapping areas, the pixel is assigned for the cell with higher
                prediction scores.
        """
        preds = self.predict(image, **kwargs)

        return patches_to_label(
            preds,
            input_size=image.shape[:2],
            score_threshold=score_threshold,
        )

    def predict_on_large_image(
        self,
        image: ArrayLike,
        gs: int,
        ss: int,
        *,
        scaling: float = 1,
        min_area: int = 0,
        nms_iou: float = 0.3,
        segmentation_threshold: float = 0.5,
        score_threshold: float = 0.5,
        return_label: bool = False,
        **kwargs,
    ) -> Union[dict, ArrayLike]:
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
            scaling: A image scaling factor.
            nms_iou: Optional iou threshold for the non-max-suppression post-processing.
                Default is 0, which disable non-max-suppression.
            min_area: Optional minimum area for the instance to be included in the results.
                Default is 0.
            segmentation_threshold: Default is 0.5.
            return_label: Whether to output full model prediction or segmentation
                label.

        Returns:
            Segmention label if output_label is True, otherwise a dict of full model
                predictions.
        """

        get_padding = lambda a, gs, ss: (a - 1) // ss * ss + gs - a

        h, w = image.shape[:2]
        padded_img = np.pad(
            image, [[0, get_padding(h, gs, ss)], [0, get_padding(w, gs, ss)], [0, 0]]
        )

        preds = []
        for y0 in range(0, h, ss):
            for x0 in range(0, w, ss):
                logging.info(f"Processing grid {y0}-{x0}")
                pred = self.predict(
                    padded_img[y0 : y0 + gs, x0 : x0 + gs],
                    scaling=scaling,
                    min_area=min_area,
                    **kwargs,
                )
                pred = _remove_edge_instances(
                    pred,
                    [gs, gs],
                    remove_top=y0 != 0,
                    remove_bottom=(y0 + gs) != h,
                    remove_left=x0 != 0,
                    remove_right=(x0 + gs) != w,
                )

                logging.info(f"Transfer result...")
                boolean_mask = np.asarray(
                    pred["instance_mask"].squeeze((1, 2))
                    & (pred["pred_scores"] >= score_threshold)
                )
                pred = dict(
                    pred_scores=np.array(pred["pred_scores"]),
                    pred_locations=np.array(
                        pred["pred_locations"] + jnp.asarray([y0, x0])
                    ),
                    pred_bboxes=np.array(
                        pred["pred_bboxes"] + jnp.asarray([y0, x0, y0, x0])
                    ),
                    pred_masks=np.array(
                        pred["instance_output"] >= segmentation_threshold
                    ),
                    pred_masks_y0=np.array(pred["instance_yc"][:, 0, 0] + y0),
                    pred_masks_x0=np.array(pred["instance_xc"][:, 0, 0] + x0),
                )
                pred = jax.tree_util.tree_map(lambda x: x[boolean_mask], pred)
                preds.append(pred)

        logging.info(f"Postprocessing...")
        preds = jax.tree_util.tree_map(lambda *x: np.concatenate(x), *preds)
        valid_locs = (
            (preds["pred_locations"] > 0).all(axis=1)
            & (preds["pred_locations"][:, 0] < h)
            & (preds["pred_locations"][:, 1] < w)
        )
        # valid_locs = valid_locs & (preds["pred_scores"] >= score_threshold)
        # valid_locs = valid_locs & (np.count_nonzero(preds["pred_masks"]) >= min_area)

        preds = jax.tree_util.tree_map(lambda x: x[valid_locs], preds)

        # sort based on scores
        asort = np.argsort(preds["pred_scores"])[::-1]
        preds = jax.tree_util.tree_map(lambda x: x[asort], preds)

        # nms
        if nms_iou > 0:
            logging.info(f"nms...")
            scores = preds["pred_scores"]
            boxes = preds["pred_bboxes"]
            _, _, selections = sorted_non_max_suppression(
                scores,
                boxes,
                -1,
                threshold=nms_iou,
                return_selection=True,
            )

            preds = jax.tree_util.tree_map(lambda x: x[selections], preds)

        if return_label:
            logging.info(f"Generating label...")

            ps = preds["pred_masks"].shape[-1]

            label = np.zeros([h + ps, w + ps], dtype=int)
            cnt = len(preds["pred_masks"])
            for mask, y0, x0 in zip(
                preds["pred_masks"][::-1],
                preds["pred_masks_y0"][::-1],
                preds["pred_masks_x0"][::-1],
            ):
                yc, xc = np.where(mask)
                label[(yc + y0 + ps // 2, xc + x0 + ps // 2)] = cnt
                cnt -= 1

            return label[ps // 2 : ps // 2 + h, ps // 2 : ps // 2 + w]

        else:
            return preds

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

    # FIXME Change the detector without first compile the mode will result
    # in error. This is a hidden gotcha. Need to setup warning to the user.
    @detector.setter
    def detector(self, new_detector):
        from .modules import Detector

        cur_module = self.module

        if isinstance(new_detector, dict):
            new_detector = Detector(**new_detector)

        if isinstance(new_detector, Detector):
            self.model = (
                lacss.modules.Lacss(
                    self.module.backbone,
                    self.module.lpn,
                    new_detector,
                    self.module.segmentor,
                ),
                self.model[1],
            )
        else:
            raise ValueError(f"{new_detector} is not a Lacss Detector")

    def save(self, save_path) -> None:
        """Save the model by pickling.
            In the form of (module, weights).

        Args:
            save_path: Path to the pkl file
        """
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
