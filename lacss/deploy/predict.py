""" 
Attributes:
    model_urls: URLs for build-in pretrain models. e.g model_urls["default"].
"""
from __future__ import annotations

import logging
import math
from functools import partial, reduce
from typing import Mapping, Sequence, Tuple

import cv2
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ..ops import patches_to_label, non_max_suppression, bboxes_of_patches, crop_and_resize_patches, box_iou_similarity
from ..utils import load_from_pretrained
from ..typing import Array, ArrayLike

Shape = Sequence[int]

model_urls: Mapping[str, str] = {
    "lacss2s-n-bf": "https://huggingface.co/jiyuuchc/lacss2s-n-bf/resolve/main/lacss2s-n-bf",
    "lacss2s-a": "https://huggingface.co/jiyuuchc/lacss2s-a/resolve/main/lacss2s-a",
    "lacss2s-b": "https://huggingface.co/jiyuuchc/lacss2s-b/resolve/main/lacss2s-b",
}
model_urls["default"] = model_urls["lacss2s-b"]

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


def _to_polygons(
    predictions, mask,
    *,
    scaling=1.0,
    segmentation_threshold=0,
    chain_approx=cv2.CHAIN_APPROX_SIMPLE,
) -> list:
    polygons = []

    y0s = np.asarray(predictions["segmentation_y0_coord"])
    x0s = np.asarray(predictions["segmentation_x0_coord"])
    segs = np.asarray(predictions["segmentations"] >= segmentation_threshold).astype("uint8")
    mask = np.asarray(mask)

    assert segs.shape[1] == 1
    segs = segs.squeeze(1)

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


def _draw_label(
    segs, y0s, x0s, img_shape, *,
    scaling=1.0,
    chain_approx=cv2.CHAIN_APPROX_SIMPLE,
):
    segs = segs.swapaxes(0,1)
    max_label = segs.shape[1]
    label = np.zeros(img_shape, dtype="uint16")

    for d, seg_slice in enumerate(segs):
        for k, (y0, x0, seg) in enumerate(zip(y0s[::-1], x0s[::-1], seg_slice[::-1])):
            c, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, chain_approx)

            if len(c) > 0:
                max_len_element = reduce(
                    lambda a, b: a if len(a) >= len(b) else b,
                    c,
                    np.zeros([0, 1, 2], dtype=int),
                )
                polygon = max_len_element.squeeze(1).astype(float) + [x0, y0]

                if scaling != 1.0:
                    polygon = (polygon + 0.5) / scaling - 0.5

                polygon = np.round(polygon).astype(int)
                cv2.fillPoly(label[d], [polygon], max_label - k)

    return label


def _compute_label(predictions, scaling, mask, image_shape, segmentation_threshold):
    if scaling == 1.0:
        return np.asarray(patches_to_label(
            predictions, 
            image_shape,
            mask=mask,
            score_threshold=0,
            threshold=segmentation_threshold,
        ))
    else:
        y0s = np.asarray(predictions["segmentation_y0_coord"])
        x0s = np.asarray(predictions["segmentation_x0_coord"])
        segs = np.asarray(predictions["segmentations"] >= segmentation_threshold).astype("uint8")
        segs, y0s, x0s = segs[mask], y0s[mask], x0s[mask]

        return _draw_label(
            segs, y0s, x0s, image_shape, scaling=scaling
        )


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
    """
    DEFAULT_IMG_SIZE=544

    def __init__(
            self, 
            url: str | tuple[nn.Module, dict],
        ):
        """Construct Predictor

        Args:
            url: A URL or local path to the saved model.
                URLs for build-in pretrained models can be found in lacss.deploy.model_urls
        """

        if isinstance(url, tuple) and len(url) == 2:
            self.model = url
            if not isinstance(self.model[0], nn.Module):
                raise ValueError(
                    "Initiaize the Predictor with a tuple, but the first element is not a Module."
                )
        else:
            self.model = load_from_pretrained(url)

        self._apply_fn = jax.jit(partial(self.module.apply, dict(params=self.params)))

        logging.info(f"Prcompile the predictor for image shape {self.DEFAULT_IMG_SIZE, self.DEFAULT_IMG_SIZE}. This may take several minutes.")

        precompile_shape = [self.DEFAULT_IMG_SIZE] * 2 + [3]
        x = jnp.zeros(precompile_shape)
        _ = self.predict(x)


    def _format_image(self, image, scale=1.0):
        from skimage.transform import rescale

        if image.ndim == 3:
            image = image[None, ...]

        if scale != 1.0:
            image = rescale(image, [1.0, scale, scale], channel_axis=-1)

        image = image - image.mean()
        image = image / image.std()

        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        D, H, W, C = image.shape
        padto_h, padto_w = (H-1)//32*32+32, (W-1)//32*32+32
        if D == 1 and padto_h < self.DEFAULT_IMG_SIZE and padto_w < self.DEFAULT_IMG_SIZE:
            padto_h, padto_w = self.DEFAULT_IMG_SIZE, self.DEFAULT_IMG_SIZE
        padding = [[0, 0], [0, padto_h-H], [0, padto_w-W], [0, 3-C]]
        image = np.pad(image, padding)

        assert image.shape[-1] == 3, f"input image has more than 3 channels"

        depth = image.shape[0]
        if depth > 1:
            z_pad = 16 - depth if depth <= 16 else 32 - depth #FIXME avoid hardcoded padding
            image = np.pad(image, [[0, z_pad], [0,0], [0,0], [0,0]])

        mask_z = np.expand_dims(np.arange(image.shape[0]) < D, (1,2))
        mask_y = np.expand_dims(np.arange(image.shape[1]) < H, (0,2))
        mask_x = np.expand_dims(np.arange(image.shape[2]) < W, (0,1))
        input_mask = mask_z & mask_y & mask_x

        return image, input_mask


    def predict(
        self,
        image: ArrayLike,
        *,
        min_area: float = 0,
        scaling: float = 1.0,
        output_type: str = "label",
        score_threshold: float = 0.5,
        segmentation_threshold: float = 0.5,
        nms_iou: float = 1,
        remove_out_of_bound: bool|None = None,
    ) -> dict:
        """Predict segmentation.

        Args:
            image: A ndarray of (h,w,c) or (d,h,w,c) format. c must be 1-3

        Keyword Args:
            output_type: "label" | "contour" | "bbox"
            min_area: Minimum area of a valid prediction.
            scaling: A image scaling factor. If not 1, the input image will be resized internally before fed
                to the model. The results will be resized back to the scale of the orginal input image.
            score_threshold: Min score needed to be included in the output.
            segmentation_threshold: Threshold value for segmentation
            nms_iou: IOU threshold value for non-max-supression post-processing

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
        if remove_out_of_bound is not None:
            import warnings
            warnings.warn("remove_out_of_bound is deprecated", DeprecationWarning, 2)

        if image.ndim == 2:
            logging.warn("Input image has a ndim of 2. Reformate assuming this is a single-channel 2D image.")
            image = image[..., None]
        is_2d_input = image.ndim == 3 
        if is_2d_input:
            image = image[None, ...]

        orig_shape = image.shape

        image, input_mask = self._format_image(image, scaling)

        if not output_type in ("bbox", "label", "contour"):
            raise ValueError(
                f"output_type should be 'bbox'|'label'|'contour'. Got {output_type} instead."
            )
        
        if output_type == "contour" and image.shape[0] != 1:
            raise ValueError(f"contour output is only supported on 2D inputs")

        seg_logit_threshold = math.log(segmentation_threshold / (1 - segmentation_threshold))

        logging.info(f"start model prediction with image {image.shape}")
        preds = self._apply_fn(image, mask=input_mask)["predictions"]
        logging.info("done model prediction")

        instance_mask = np.asarray(preds["segmentation_is_valid"])
        instance_mask &= preds["scores"] >= score_threshold

        # mask-off invalid pixels
        # patch_size = preds['segmentations'].shape[-1]
        # mg = jnp.expand_dims(jnp.mgrid[:patch_size, :patch_size], 1) # [2, 1, ps, ps]
        # y0x0 = jnp.stack([preds['segmentation_y0_coord'], preds['segmentation_x0_coord']])
        # mg = jnp.moveaxis(mg + y0x0[:, :, None, None], 0, -1) #[N, ps, ps, 2]
        # mg = jnp.expand_dims(mg, 1) # [N, 1, ps, ps, 2]
        # valid = (mg > 0).all(axis=-1) & (mg < jnp.array(orig_shape[-3:-1])).all(axis=-1) #[N, 1, ps, ps]
        # valid = valid & input_mask #[N, d, ps, ps]
        # preds['segmentations'] = jnp.where(
        #     valid, preds['segmentations'], -1e8
        # )

        if min_area > 0:
            areas = jnp.count_nonzero(
                preds["segmentations"] > segmentation_threshold, axis=(1, 2, 3)
            )
            instance_mask &= areas >= min_area * scaling * scaling

        scores = preds["scores"]
        bboxes = bboxes_of_patches(preds, threshold=seg_logit_threshold) / scaling

        if nms_iou > 0 and nms_iou < 1:
            _, _, sel = non_max_suppression(
                scores, bboxes, bboxes.shape[0],
                threshold=nms_iou,
                return_selection=True,
                similarity_func=box_iou_similarity,
            )
            instance_mask = instance_mask & sel
        
        if output_type == "bbox":
            target_shape = (48, 48) if len(orig_shape) < 4 else (8, 48, 48)
            segmentations = crop_and_resize_patches(
                preds, target_shape, bboxes,
                convert_logits=True,
            )
            return dict(
                pred_scores=np.asarray(scores)[instance_mask],
                pred_bboxes=np.asarray(bboxes)[instance_mask],
                pred_masks=np.asarray(segmentations)[instance_mask],
            )

        elif output_type == "contour":
            contours = _to_polygons(
                preds, instance_mask,
                scaling=scaling,
                segmentation_threshold=seg_logit_threshold,
            )
            return dict(
                pred_scores=np.asarray(scores)[instance_mask],
                pred_bboxes=np.asarray(bboxes)[instance_mask],
                pred_contours=contours,
            )

        else:  # Label
            label = _compute_label(
                preds, scaling, instance_mask, 
                image.shape[:1] + orig_shape[-3:-1],
                seg_logit_threshold,
            )
            label = label[:orig_shape[0]]
            if is_2d_input:
                label = label.squeeze(0)
            return dict(
                pred_scores=np.asarray(scores)[instance_mask],
                pred_label=label,
            )


    # def predict_on_large_image(
    #     self,
    #     image: ArrayLike,
    #     gs: int,
    #     ss: int,
    #     *,
    #     scaling: float = 1,
    #     min_area: int = 0,
    #     nms_iou: float = 0.25,
    #     segmentation_threshold: float = 0.5,
    #     score_threshold: float = 0.5,
    #     output_type: str = "label",
    #     min_cells_per_patch: int = 0,
    #     disable_padding: bool = False,
    #     **kwargs,
    # ) -> dict:
    #     """Make prediction on very large image by dividing into a grid.

    #     Direct model prediction on large image may cause out-of-memory error. This
    #     method divided the large image to smaller grid and then stitch the results
    #     to form the complete prediction.

    #     Args:
    #         image: An image with (H, W, C) format.
    #         gs: An int value. Grid size of the computation.
    #         ss: An int value of stepping size. Must be small than gs to produce valid
    #             results.

    #     Keyword Args:
    #         output_type: label" | "contour" | "bbox"
    #         min_area: Minimum area of a valid prediction.
    #         scaling: A image scaling factor. If not 1, the input image will be resized internally before fed
    #             to the model. The results will be resized back to the scale of the orginal input image.
    #         score_threshold: Min score needed to be included in the output.
    #         segmentation_threshold: Threshold value for segmentation.
    #         nms_iou: IOU threshold value for non-max-supression durint post-processing

    #     Returns:
    #         For "label" output:

    #             - pred_scores: The prediction scores of each instance.
    #             - pred_label: a 2D image label. 0 is background.

    #         For "contour" output:

    #             - pred_scores: The prediction scores of each instance.
    #             - pred_contours: a list of polygon arrays in x-y format.

    #         For "bbox" output (ie MaskRCNN):

    #             - pred_scores: The prediction scores of each instance.
    #             - pred_bboxes: The bounding-boxes of detected instances in y0x0y1x1 format
    #             - pred_masks: A 3d array representing segmentation within bboxes
    #     """

    #     if not output_type in ("bbox", "label", "contour"):
    #         raise ValueError(
    #             f"output_type should be 'patch'|'label'|'contour'. Got {output_type} instead."
    #         )

    #     module, params = self.model

    #     if len(kwargs) > 0:
    #         apply_fn = _cached_partial(module.apply, **kwargs)
    #     else:
    #         apply_fn = module.apply

    #     inv_sig_st = math.log(segmentation_threshold / (1 - segmentation_threshold))
    #     ctx = freeze(
    #         dict(
    #             apply_fn=apply_fn,
    #             output_type="bbox" if output_type == "bbox" else "grid",
    #             segmentation_threshold=inv_sig_st,
    #             score_threshold=score_threshold,
    #             scaling=scaling,
    #             min_area=min_area,
    #             remove_out_of_bound=True,
    #             nms_iou=0,
    #         )
    #     )

    #     h, w = image.shape[:2]

    #     if not disable_padding:
    #         get_padding = lambda a, gs, ss: (a - 1) // ss * ss + gs - a

    #         padded_img = np.pad(
    #             image,
    #             [[0, get_padding(h, gs, ss)], [0, get_padding(w, gs, ss)], [0, 0]],
    #         )
    #     else:
    #         padded_img = image

    #     preds = []
    #     for y0 in range(0, h, ss):
    #         for x0 in range(0, w, ss):
    #             logging.info(f"Processing grid {y0}-{x0}")
    #             pred = _predict(
    #                 ctx,
    #                 params,
    #                 padded_img[y0 : y0 + gs, x0 : x0 + gs],
    #             )
    #             pred = _remove_edge_instances(
    #                 pred,
    #                 [gs, gs],
    #                 remove_top=y0 > 0,
    #                 remove_bottom=(y0 + gs) < h,
    #                 remove_left=x0 > 0,
    #                 remove_right=(x0 + gs) < w,
    #                 threshold=segmentation_threshold,
    #             )

    #             logging.info(f"Transfer result...")

    #             mask = np.asarray(pred["instance_mask"])

    #             if np.count_nonzero(mask) > min_cells_per_patch:
    #                 if output_type != "bbox":
    #                     pred = dict(
    #                         scores=np.array(pred["scores"])[mask],
    #                         bboxes=np.array(pred["bboxes"])[mask] + [y0, x0, y0, x0],
    #                         segmentations=np.array(pred["segmentations"])[mask],
    #                         y0s=np.array(pred["y0s"])[mask] + y0,
    #                         x0s=np.array(pred["x0s"])[mask] + x0,
    #                     )

    #                     polygons = _to_polygons(
    #                         pred,
    #                         segmentation_threshold=segmentation_threshold,
    #                         scaling=scaling,
    #                     )

    #                     valid_polygons = np.asarray([len(p) > 0 for p in polygons])
    #                     scores = np.where(valid_polygons, pred["scores"], -1)
    #                     bboxes = np.where(
    #                         valid_polygons[:, None], pred["bboxes"] / scaling, -1
    #                     )

    #                     preds.append(
    #                         dict(
    #                             scores=scores,
    #                             polygons=np.fromiter(polygons, dtype=object),
    #                             bboxes=bboxes,
    #                         )
    #                     )

    #                 else:
    #                     bboxes = np.array(pred["bboxes"])[mask] + [y0, x0, y0, x0]
    #                     bboxes /= scaling
    #                     preds.append(
    #                         dict(
    #                             scores=np.array(pred["scores"])[mask],
    #                             bboxes=bboxes,
    #                             segmentations=np.array(pred["segmentations"])[mask],
    #                         )
    #                     )

    #     logging.info(f"Postprocessing...")
    #     preds = jax.tree_util.tree_map(lambda *x: np.concatenate(x), *preds)

    #     # sort based on scores
    #     asort = np.argsort(preds["scores"])[::-1]
    #     preds = jax.tree_util.tree_map(lambda x: x[asort], preds)

    #     # nms
    #     if nms_iou > 0:
    #         logging.info(f"nms...")
    #         scores = preds["scores"]
    #         boxes = preds["bboxes"]
    #         _, _, selections = non_max_suppression(
    #             scores,
    #             boxes,
    #             -1,
    #             threshold=nms_iou,
    #             min_score=0,
    #             return_selection=True,
    #         )

    #         preds = jax.tree_util.tree_map(lambda x: x[selections], preds)

    #     if output_type == "bbox":  # FIXME unimplemented

    #         return dict(
    #             pred_scores=preds["scores"],
    #             pred_bboxes=preds["bboxes"],
    #             pred_masks=segs,
    #         )

    #     if output_type == "label":
    #         logging.info(f"Generating label...")

    #         label = _draw_label(preds["polygons"], image.shape[:2])

    #         return dict(
    #             pred_scores=preds["scores"],
    #             pred_label=label,
    #         )

    #     else:  # contours
    #         return dict(
    #             pred_scores=preds["scores"],
    #             pred_contours=preds["polygons"],
    #         )

    @property
    def module(self) -> nn.Module:
        return self.model[0]

    @property
    def params(self) -> dict:
        return self.model[1]

    @params.setter
    def params(self, new_params):
        self.model = self.module, new_params
