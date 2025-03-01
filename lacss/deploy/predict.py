""" 
Attributes:
    model_urls: URLs for build-in pretrain models. e.g model_urls["default"].
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from functools import partial, reduce
from typing import Any, Mapping, Sequence, Tuple

import cv2
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from vedo import Volume

from ..modules import Lacss
from ..ops import bboxes_of_patches, crop_and_resize_patches
from ..typing import Array, ArrayLike
from ..utils import load_from_pretrained

Shape = Sequence[int]

logger = logging.getLogger(__name__)


def _ensemble_model_fn(module, params, image) -> dict:
    """A JIT compiled function to call Lacss model ensemble
    if params is a sequence (not a dict), assuming a model ensemble and
    generate ensembed prediction.
    """
    from lacss.modules.lpn import generate_predictions

    model_ = module.copy()
    model_.segmentor = None
    model_.segmentor_3d = None

    lpn_out, seg_features = [], []
    for p in params:
        model_out = jax.jit(model_.apply)(dict(params=p), image)

        lpn_out.append(model_out["detector"])

        seg_features.append(model_out["seg_features"])

    lpn_out = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *lpn_out)

    lpn_out = dict(
        logits=lpn_out["logits"].mean(axis=0),
        regressions=lpn_out["regressions"].mean(axis=0),
        ref_locs=lpn_out["ref_locs"][0],
    )

    lpn_out["pred_locs"] = (
        lpn_out["ref_locs"] + lpn_out["regressions"] * module.detector.feature_scale
    )

    predictions = generate_predictions(module.detector, lpn_out)

    assert len(seg_features) == len(params)

    seg_predictions = []
    if image.ndim == 3 and module.segmentor is not None:
        for x, p in zip(seg_features, params):
            seg_predictions.append(
                jax.jit(module.segmentor.apply)(
                    dict(params=p["segmentor"]), x, predictions["locations"]
                )["predictions"]
            )

    elif image.ndim == 4 and module.segmentor_3d is not None:
        seg_predictions = []
        for x, p in zip(seg_features, params):
            seg_predictions.append(
                jax.jit(module.segmentor_3d.apply)(
                    dict(params=p["segmentor_3d"]), x, predictions["locations"]
                )["predictions"]
            )

    if len(seg_predictions) > 0:
        predictions["segmentations"] = jnp.mean(
            jnp.array([x["segmentations"] for x in seg_predictions]),
            axis=0,
        )

        seg_prediction = seg_predictions[0]
        del seg_prediction["segmentations"]

        predictions.update(seg_prediction)

    return predictions


@dataclass
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

    Args:
        url: Model file (local or remote) or (module, parameters) tuple. If
            the parameters are a sequence instead of a dictionary, treat it
            as a model ensemble.
        f16: If true, compute in f16 precision (instead of the default f32)
        grid_size: Grid size for large images. Large image are broken down
            into grids to avoid GPU memeory overflow.
        step_size: Should be slightly smaller than grid_size to allow some
            overlap between grids.
        grid_size_3d: Grid_size for 3D input.
        step_size_3d: Step_size for 3D input.
        mask_size: Only use for bbox output. The dimension of the segmentation
            mask output.
        mc_step_size: Only used for 3D contour output. The step size during the
            mesh generation.
        chain_approx: Only used for 2D contour output. Polygon generation rule.
    """

    url: str | os.PathLike | tuple[Lacss, dict]
    f16: bool = False
    grid_size: int = 544
    step_size: int = 480
    cells_per_grid: int = 256
    grid_size_3d: int = 256
    step_size_3d: int = 192
    cells_per_grid_3d: int = 256
    mask_size: int = 36
    mc_step_size: int = 1
    chain_approx: int = cv2.CHAIN_APPROX_SIMPLE

    def __post_init__(self):
        """validate parameters"""

        if isinstance(self.url, tuple):
            module, params = self.url

        else:
            module, params = load_from_pretrained(self.url)

        assert (
            type(module) == Lacss
        ), f"Loaded module is not Lacss, but of {type(module)}"

        # if isinstance(params, Mapping):
        # params = [params]

        # default model config during inference
        module.detector.max_output = self.cells_per_grid

        if module.detector_3d:
            module.detector_3d.max_output = self.cells_per_grid_3d

        if module.segmentor:
            module.segmentor.full_scale_output = True

        if module.segmentor_3d:
            module.segmentor_3d.full_scale_output = False

        self.module = module
        self.params = params

        if self.f16:
            self.params = jax.tree_util.tree_map(
                lambda x: x.astype("float16"),
                self.params,
            )

        assert (
            self.step_size < self.grid_size
        ), f"step_size ({self.step_size}) not smaller than grid_size ({self.grid_size})"
        assert (
            self.grid_size % 32 == 0
        ), f"grid_size ({self.grid_size}) is not divisable by 32"
        assert (
            self.step_size_3d < self.grid_size_3d
        ), f"step_size ({self.step_size_3d}) not smaller than grid_size ({self.grid_size_3d})"
        assert (
            self.grid_size_3d % 32 == 0
        ), f"grid_size ({self.grid_size_3d}) is not divisable by 32"
        assert (
            self.cells_per_grid % 16 == 0
        ), f"cells_per_grid ({self.cells_per_grid}) is not divisable by 16"
        assert (
            self.cells_per_grid_3d % 16 == 0
        ), f"cells_per_grid_3d ({self.cells_per_grid_3d}) is not divisable by 16"

        self._model_fn = jax.jit(
            self.module.apply
        )  # FIXME this locks down model hyperparameter

    def _resize_image(self, image, target_shape):
        from skimage.transform import resize

        # process the data in float32 to avoid overflow
        # only convert to float16 (if needed) at the end

        image = image.astype("float32")

        image = image / (image.std() + 1e-4)
        padding_value = image.mean()
        image -= padding_value

        orig_shape = image.shape[:-1]

        if target_shape is None:
            target_shape = np.array(orig_shape)

        else:
            target_shape = np.broadcast_to(target_shape, [len(orig_shape)])

            scaling = target_shape / orig_shape

            if np.all(np.abs(scaling - 1) < 0.1):  # ignore small scaling
                target_shape = np.array(orig_shape)

            else:
                image = resize(image, target_shape)

        if self.f16:
            image = image.astype("float16")

        logger.debug(f"resized image data from {orig_shape} to {target_shape}")

        return image, target_shape, padding_value

    def _compute_contours_2d(self, predictions, img_sz, threshold):
        """2D contour using cv2"""
        mask = np.array(predictions["segmentation_is_valid"])

        y0s = np.asarray(predictions["segmentation_y0_coord"])
        x0s = np.asarray(predictions["segmentation_x0_coord"])
        segs = np.asarray(predictions["segmentations"] >= threshold).astype("uint8")

        segs = segs.squeeze(1)

        polygons, bboxes = [], []
        for k in range(segs.shape[0]):
            if mask[k]:
                c, _ = cv2.findContours(segs[k], cv2.RETR_EXTERNAL, self.chain_approx)
                max_len_element = reduce(
                    lambda a, b: a if len(a) >= len(b) else b,
                    c,
                    np.zeros([0, 1, 2], dtype=int),
                )

                polygon = max_len_element.squeeze(1).astype(float) + 0.5

                polygon += [x0s[k], y0s[k]]

                if len(polygon) == 0:
                    mask[k] = False
                else:
                    box = np.r_[polygon.min(axis=0), polygon.max(axis=0)]
                    box = box[[1, 0, 3, 2]]
                    if (box[:2] >= img_sz).any() or (box[2:] <= 0).any():
                        mask[k] = False
                    else:
                        polygons.append(polygon)
                        bboxes.append(box)

        assert np.count_nonzero(mask) == len(polygons)

        assert np.count_nonzero(mask) == len(bboxes)

        scores = np.array(predictions["scores"])[mask]

        bboxes = np.array(bboxes).reshape(-1, 4)

        return scores, polygons, bboxes

    def _compute_contours_3d(self, predictions, img_sz, threshold):
        meshes, bboxes = [], []

        mask = np.array(predictions["segmentation_is_valid"])

        z0s = np.asarray(predictions["segmentation_z0_coord"])
        y0s = np.asarray(predictions["segmentation_y0_coord"])
        x0s = np.asarray(predictions["segmentation_x0_coord"])
        segs = np.asarray(predictions["segmentations"] > threshold)

        assert segs.ndim == 4

        for k in range(segs.shape[0]):
            if mask[k]:
                mesh = (
                    Volume(segs[k], spacing=[2, 2, 2])  # model output is 2x binned
                    .isosurface(value=0.5, flying_edges=True)
                    .shift(z0s[k], y0s[k], x0s[k])
                )

                box = np.r_[mesh.bounds()[::2], mesh.bounds()[1::2]]

                if (box[:3] >= img_sz).any() or (box[3:] <= 0).any():
                    mask[k] = False

                else:
                    meshes.append(mesh)
                    bboxes.append(box)

        assert np.count_nonzero(mask) == len(meshes)

        assert np.count_nonzero(mask) == len(bboxes)

        scores = np.array(predictions["scores"])[mask]

        bboxes = np.array(bboxes).reshape(-1, 6)

        return scores, meshes, bboxes

    def _compute_contours(self, predictions, img_sz, threshold):
        if len(img_sz) == 3:
            return self._compute_contours_3d(predictions, img_sz, threshold)
        else:
            return self._compute_contours_2d(predictions, img_sz, threshold)

    def _call_model(
        self, patch, score_threshold, size_threshold, threshold, padding_value
    ):
        """Core inference routine:
        1. Pad input image to regular size and call model function
        2. clean up results
        3. compute contours
        """
        logger.debug(f"process image patch of {patch.shape}")

        # pad image to fixed input sizes
        patch_sz = patch.shape[:-1]
        dim = patch.ndim - 1

        if dim == 2:  # 2D input
            assert (
                max(patch_sz) <= self.grid_size
            ), f"attempted to call model with image shape ({patch_sz}) that exceeded limit."
            h, w = patch_sz
            padding_shape = [
                [0, self.grid_size - h],
                [0, self.grid_size - w],
                [0, 0],
            ]

        else:
            assert (
                max(patch_sz) <= self.grid_size_3d
            ), f"attempted to call model with image shape exceed limit."
            d, h, w = patch_sz
            padding_shape = [
                [0, self.grid_size_3d - d],
                [0, self.grid_size_3d - h],
                [0, self.grid_size_3d - w],
                [0, 0],
            ]

        patch = np.pad(patch, padding_shape, constant_values=padding_value)

        # make 3-ch
        if patch.shape[-1] == 1:
            patch = np.repeat(patch, 3, axis=-1)

        elif patch.shape[-1] == 2:
            patch = np.stack([patch, np.zeros_like(patch[..., :1])], axis=-1)

        logger.debug(f"pad patch to shape: {patch.shape}")

        if isinstance(self.params, Mapping):
            predictions = self._model_fn(dict(params=self.params), patch)["predictions"]
        else:  # handle ensemble
            predictions = _ensemble_model_fn(self.module, self.params, patch)[
                "predictions"
            ]

        # clean up
        mask = predictions["segmentation_is_valid"]

        logger.debug(f"obtained model prediction of {np.count_nonzero(mask)} cells")

        mask &= predictions["scores"] >= score_threshold
        if size_threshold > 0:
            areas = jnp.count_nonzero(
                predictions["segmentations"] > threshold,
                axis=(1, 2, 3),
            )

            mask &= areas >= size_threshold

        predictions["segmentation_is_valid"] = mask

        logger.debug(f"found {np.count_nonzero(mask)} cells after clean up")

        # additional computation
        scores, contours, bboxes = self._compute_contours(
            predictions, patch_sz, threshold
        )

        logger.debug(f"done converting results to contours.")

        assert scores.shape[0] == len(bboxes)
        assert scores.shape[0] == len(contours)

        return scores, contours, bboxes

    def _mark_edge_instances(self, bboxes, pos, patch_sz, img_sz):
        bboxes = np.array(bboxes)
        removal = np.zeros([bboxes.shape[0]], dtype=bool)

        dim = len(img_sz)

        removal |= ((bboxes[:, :dim] <= 1) & (pos > 0)).any(axis=-1)
        removal |= ((bboxes[:, dim:] + 1 >= patch_sz) & (pos + patch_sz < img_sz)).any(
            axis=-1
        )

        return ~removal

    def _process_image(
        self, image, score_threshold, size_threshold, threshold, padding_value
    ):
        """process a 2d/3d image in grids"""
        img_sz = image.shape[:-1]
        img_dim = len(img_sz)

        if img_dim == 2:
            gs, ss = (self.grid_size, self.step_size)
        else:
            gs, ss = (self.grid_size_3d, self.step_size_3d)

        grid_positions = [slice(0, max(d - (gs - ss), 1), ss) for d in img_sz]
        grid_positions = np.moveaxis(np.mgrid[grid_positions], 0, -1)
        grid_positions = grid_positions.reshape(-1, img_dim)

        is_single_grid = grid_positions.shape[0] == 1

        all_scores, all_contours, all_bboxes, all_masks = [], [], [], []
        for pos in grid_positions:
            logger.debug(f"patch position is {pos}")

            slices = (slice(x, x + gs) for x in pos)
            patch = image.__getitem__(tuple(slices))
            patch_sz = patch.shape[:-1]

            scores, contours, bboxes = self._call_model(
                patch,
                score_threshold,
                size_threshold,
                threshold,
                padding_value,
            )

            logger.debug(f"processing patch results")

            # mask = self._mark_edge_instances(bboxes, pos, patch_sz, img_sz)
            mask = np.ones([bboxes.shape[0]], dtype=bool)

            bboxes += np.r_[pos, pos]

            if img_dim == 3:
                contours = [c.shift(pos[0], pos[1], pos[2]) for c in contours]

            else:
                contours = [p + pos[::-1] for p in contours]

            all_scores.append(scores)
            all_contours.append(contours)
            all_bboxes.append(bboxes)
            all_masks.append(mask)

        logger.debug(f"concatenate all results")

        scores = np.concatenate(all_scores)

        if is_single_grid and len(self.params) == 1:
            asort = None
        else:
            asort = np.argsort(scores)[::-1]

        return dict(
            scores=scores,
            mask=np.concatenate(all_masks),
            bboxes=np.concatenate(all_bboxes),
            contours=sum(all_contours, []),
            asort=asort,
        )

    def _non_max_supression(self, predictions, nms_iou):
        from lacss.ops import box_iou_similarity

        if nms_iou <= 0 or nms_iou >= 1:
            return

        selected = predictions["mask"]
        asort = predictions["asort"]
        boxes = predictions["bboxes"]

        if asort is not None:
            boxes = boxes[asort]
            selected = selected[asort]

        sm = np.array(box_iou_similarity(boxes, boxes))
        sm = np.triu(sm, k=1)

        assert len(selected) == boxes.shape[0]

        for k in range(len(selected)):
            if selected[k]:
                selected &= sm[k] < nms_iou

        # inverse selected` to the original order
        if asort is not None:
            selected = selected[asort.argsort()]

        predictions["mask"] = selected

    def _post_process(self, predictions, scaling, nms_iou):
        self._non_max_supression(
            predictions=predictions,
            nms_iou=nms_iou,
        )

        mask = predictions["mask"]

        scores = predictions["scores"][mask]

        bboxes = predictions["bboxes"][mask] / np.r_[scaling, scaling]

        if len(scaling) == 2:
            contours = [
                p / scaling[::-1] for p, bm in zip(predictions["contours"], mask) if bm
            ]

        else:
            contours = []
            for mesh, bm in zip(predictions["contours"], mask):
                if bm:
                    mesh = mesh.scale(1 / scaling, origin=False)
                    divisions = np.ceil(
                        (mesh.bounds()[1::2] - mesh.bounds()[::2]) / self.mc_step_size
                    ).astype(int)
                    mesh = mesh.decimate_binned(divisions)

                    contours.append(mesh)

        return dict(
            scores=scores,
            bboxes=bboxes,
            contours=contours,
        )

    def _to_contour_format(self, preds, img_sz):
        if len(img_sz) == 3:
            simple_contours = []
            for mesh in preds["contours"]:
                simple_contours.append(
                    dict(
                        verts=mesh.vertices,
                        faces=np.array(mesh.cells),
                    )
                )
            preds["contours"] = simple_contours

        return dict(
            pred_scores=preds["scores"],
            pred_bboxes=preds["bboxes"],
            pred_contours=preds["contours"],
        )

    def _to_label_format(self, preds, img_sz):
        label = np.zeros(img_sz, dtype="uint16")

        if len(img_sz) == 2:
            polygons = preds["contours"]

            color = len(polygons)
            for polygon in polygons[::-1]:
                cv2.fillPoly(label, [polygon.astype(int)], color)  # type: ignore
                color -= 1

        else:
            meshes = preds["contours"]

            d, h, w = img_sz

            color = 1
            for mesh in meshes[::-1]:
                origin = np.floor(mesh.bounds()[::2]).astype(int)
                origin = np.clip(origin, 0, (d - 1, h - 1, w - 1))
                max_size = np.array(label.shape) - origin

                vol = mesh.binarize(
                    values=(color, 0),
                    spacing=[1, 1, 1],
                    origin=origin,
                )

                vol_d = vol.tonumpy()[: max_size[0], : max_size[1], : max_size[2]]
                size = tuple(vol_d.shape)

                region = label[
                    origin[0] : origin[0] + size[0],
                    origin[1] : origin[1] + size[1],
                    origin[2] : origin[2] + size[2],
                ]
                region[...] = np.maximum(region, vol_d)

                color = color + 1

        return dict(
            pred_scores=preds["scores"],
            pred_label=label,
        )

    def _to_bbox_format(self, preds, img_sz):
        n_det = len(preds["contours"])

        if len(img_sz) == 2:
            masks = np.zeros([n_det, self.mask_size, self.mask_size], dtype="uint8")

            for k in range(n_det):
                p = preds["contours"][k]
                box = preds["bboxes"][k]
                p = (p - box[:2]) / (box[2:] - box[:2]) * self.mask_size
                cv2.fillPoly(masks[k], [p.astype(int)], 255)  # type: ignore

            return dict(
                pred_scores=preds["scores"],
                pred_bboxes=preds["bbxoxes"],
                pred_masks=masks,
            )

        else:
            masks = []

            for k in range(n_det):
                mesh = preds["contours"][k]
                box = preds["bboxes"][k]

                masks.append(
                    mesh.binarize(
                        values=(255, 0),
                        dims=[self.mask_size] * 3,
                        origin=box[:3],
                    )
                    .tonumpy()
                    .astype("uint8")
                )

            return dict(
                pred_scores=preds["scores"],
                pred_bboxes=preds["bboxes"],
                pred_masks=np.stack(masks),
            )

    def predict(
        self,
        image: ArrayLike,
        *,
        output_type: str = "label",
        reshape_to: int | tuple[int] | np.ndarray | None = None,
        min_area: float = 0,
        score_threshold: float = 0.5,
        segmentation_threshold: float = 0.5,
        nms_iou: float = 0.4,
        remove_out_of_bound: bool | None = None,
    ) -> dict:
        """Predict segmentation.

        Args:
            image: A ndarray of (h,w,c) or (d,h,w,c) format. c must be 1-3

        Keyword Args:
            output_type: "label" | "contour" | "bbox"
            reshape_to: If not None, the input image will be resized internally before send
                to the model. The results will be resized back to the scale of the orginal input image.
            min_area: Minimum area of a valid prediction.
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
        image = np.asarray(image)

        logger.debug(f"started prediction with image {image.shape}")
        start_time = time.time()

        if remove_out_of_bound is not None:
            import warnings

            warnings.warn("remove_out_of_bound is deprecated", DeprecationWarning, 2)

        if image.ndim == 2 or image.shape[-1] > 3:
            logger.warning("input seems to have no channel dim. Add one")
            image = image[..., None]

        assert image.ndim == 3 or image.ndim == 4, f"illegal image dim {image.shape}"

        assert image.shape[-1] <= 3, f"input image has more than 3 channels"

        if not output_type in ("bbox", "label", "contour", "_raw"):
            raise ValueError(
                f"output_type should be 'bbox'|'label'|'contour'. Got {output_type} instead."
            )

        img_shape = image.shape[:-1]
        dim = len(img_shape)

        image, reshape_to, padding_value = self._resize_image(image, reshape_to)

        scaling = reshape_to / img_shape

        logger.debug(f"done preprocessing")

        predictions = self._process_image(
            image,
            score_threshold,
            min_area * np.prod(scaling),
            math.log(segmentation_threshold / (1 - segmentation_threshold)),
            padding_value,
        )

        logger.debug(f"post processing ...")
        predictions = self._post_process(predictions, scaling, nms_iou)

        # generate outputs
        if output_type == "_raw":
            outputs = predictions

        elif output_type == "contour":
            outputs = self._to_contour_format(predictions, img_shape)

        elif output_type == "bbox":
            outputs = self._to_bbox_format(predictions, img_shape)

        else:
            outputs = self._to_label_format(predictions, img_shape)

        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"done prediction in {elapsed:.2f} ms")

        return outputs
