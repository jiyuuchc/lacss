""" 
Attributes:
    model_urls: URLs for build-in pretrain models. e.g model_urls["default"].
"""
from __future__ import annotations

import logging
import math
import time
from functools import reduce
from typing import Mapping, Sequence, Tuple

import cv2
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ..ops import patches_to_label, non_max_suppression, bboxes_of_patches, crop_and_resize_patches, box_iou_similarity, sub_pixel_samples
from ..utils import load_from_pretrained
from ..typing import Array, ArrayLike
from ..data.utils import image_standardization

Shape = Sequence[int]

def _remove_edge_instances(
    pred, pos, img_sz,
    gs: int = 1024+64,
):
    bboxes = pred['pred_bboxes']
    removal = np.zeros([bboxes.shape[0]], dtype=bool)

    dim = len(img_sz)

    removal |= ((bboxes[:, :dim] <= 0) & (pos > 0)).any(axis=-1)
    removal |= ((bboxes[:, dim:] >= gs) & (pos + gs < img_sz)).any(axis=-1)

    pred = {k:v[~removal] for k, v in pred.items()}

    return pred

def _get_contour(mask, chain_approx=cv2.CHAIN_APPROX_SIMPLE):
    """ returns an edge index polygon representing the outer contour """
    c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, chain_approx)
    max_len_element = reduce(
        lambda a, b: a if len(a) >= len(b) else b,
        c,
        np.zeros([0, 1, 2], dtype=int),
    )
    return max_len_element.squeeze(1).astype(float) + .5
    
def _to_polygons(
    predictions, mask,
    *,
    segmentation_threshold=0.0,
    chain_approx=cv2.CHAIN_APPROX_SIMPLE,
) -> list:
    polygons = []

    y0s = np.asarray(predictions["segmentation_y0_coord"])
    x0s = np.asarray(predictions["segmentation_x0_coord"])
    segs = np.asarray(predictions["segmentations"] >= segmentation_threshold).astype("uint8")
    mask = np.array(mask) 

    assert segs.shape[1] == 1
    segs = segs.squeeze(1)

    for k in range(segs.shape[0]):
        if mask[k]:
            polygon = _get_contour(segs[k], chain_approx) + [x0s[k], y0s[k]]
            if len(polygon) > 0:
                polygons.append(polygon)
            else:
                mask[k] = False

    return polygons, mask


def _get_mesh(seg):
    from skimage.measure import marching_cubes
    try:
        verts, faces, norms, _ = marching_cubes(seg, allow_degenerate=False)
        return dict(verts=verts, faces=faces, norms=norms)
    except:
        return None

def _to_mesh(
    predictions, mask,
    *,
    segmentation_threshold=0,
):
    from skimage.measure import marching_cubes
    meshes = []

    z0s = np.asarray(predictions["segmentation_z0_coord"])
    y0s = np.asarray(predictions["segmentation_y0_coord"])
    x0s = np.asarray(predictions["segmentation_x0_coord"])
    segs = np.asarray(predictions["segmentations"] >= segmentation_threshold).astype("uint8")
    mask = np.array(mask) 

    for k in range(segs.shape[0]):
        if mask[k]:
            mesh = _get_mesh(segs[k])
            if mesh:
                mesh['verts'] += [z0s[k], y0s[k], x0s[k]]
                meshes.append(mesh)
            else:
                mask[k] = False

    return meshes, mask  

def _format_image(image, target_shape, normalize):
    from skimage.transform import resize

    if image.shape[:-1] != tuple(target_shape):
        image = resize(image, target_shape)

    is2d = image.ndim == 3

    if normalize:
        image = image - image.mean()
        image = image / (image.std() + 1e-8)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    assert image.shape[-1] <= 3, f"input image has more than 3 channels"

    padding_shape = (target_shape - 1) // 32 * 32 + 32

    if is2d:
        if (padding_shape <=  544).all():
            padding_shape[:] = 544
        elif (padding_shape <=  1088).all():
            padding_shape[:] = 1088
    else:
        padding_shape[:] = padding_shape.max()
        if (padding_shape <  256).all():
            padding_shape[:] = 256
        else:
            padding_shape[:] = padding_shape.max()
    
    padding = [ [0, s-s0] for s, s0 in zip(padding_shape, image.shape[:-1])]
    padding += [[0, 3-image.shape[-1]]]
    image = np.pad(image, padding)

    return image


def _nms(preds, th):
    from lacss.ops import box_iou_similarity
    scores = preds["pred_scores"]
    asort = np.argsort(scores)[::-1]
    preds = {k: v[asort] for k, v in preds.items()}

    boxes = preds['pred_bboxes']
    sm = box_iou_similarity(boxes, boxes)
    for k in range(sm.shape[0]):
        sm[k, :k+1] = -1

    supressed = np.zeros([sm.shape[0]], dtype=bool)
    for k in range(sm.shape[0]):
        if not supressed[k]:
            supressed |= sm[k] > th
    
    preds = {k: v[~supressed] for k, v in preds.items()}

    return preds

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
        
        self.apply_fn = jax.jit(self.module.apply)

    def predict(
        self,
        image: ArrayLike,
        *,
        output_type: str = "label",
        reshape_to: int|tuple[int]|None = None,
        min_area: float = 0,
        score_threshold: float = 0.5,
        segmentation_threshold: float = 0.5,
        nms_iou: float = 1,
        normalize: bool = True,
        remove_out_of_bound: bool|None = None,
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
        
        logging.debug(f"started prediction with image {image.shape}")
        start_time = time.time()

        if remove_out_of_bound is not None:
            import warnings
            warnings.warn("remove_out_of_bound is deprecated", DeprecationWarning, 2)

        if image.ndim == 2 or image.shape[-1] > 3:
            logging.warning("input seems to have no channel dim. Add one")
            image = image[..., None]

        orig_shape = image.shape[:-1]
        assert image.ndim == 3 or image.ndim == 4, f"illegal image dim {image.shape}"
        is_3d = image.ndim == 4

        if not output_type in ("bbox", "label", "contour", "_raw"):
            raise ValueError(
                f"output_type should be 'bbox'|'label'|'contour'. Got {output_type} instead."
            )

        # if output_type == "contour" and image.ndim == 4:
        #     raise ValueError(f"contour output is only supported on 2D inputs")
        
        if reshape_to is None: 
            reshape_to = np.array(orig_shape)
        else:
            reshape_to = np.broadcast_to(reshape_to, [len(orig_shape)])
        scaling = reshape_to / orig_shape

        image = _format_image(image, reshape_to, normalize=normalize)

        seg_logit_threshold = math.log(segmentation_threshold / (1 - segmentation_threshold))

        logging.debug(f"done preprocessing")

        preds = self.apply_fn(dict(params=self.params), image)["predictions"]

        scores = preds["scores"]
        bboxes = bboxes_of_patches(
            preds, 
            threshold=seg_logit_threshold, 
            image_shape=reshape_to,
        )

        elapsed = (time.time() - start_time) * 1000
        logging.debug(f"finished model execution in {elapsed:.2f} ms")

        # mark all invalid instances
        instance_mask = np.asarray(preds["segmentation_is_valid"])
        instance_mask &= scores >= score_threshold
        # instance_mask &= (bboxes >= 0).all(axis=-1)
        if min_area > 0:
            areas = jnp.count_nonzero(
                preds["segmentations"] > segmentation_threshold, axis=(1, 2, 3)
            )
            instance_mask &= areas >= min_area * np.prod(scaling)
        if nms_iou > 0 and nms_iou < 1:
            sel = non_max_suppression(
                scores, bboxes, bboxes.shape[0],
                threshold=nms_iou,
                return_selection=True,
                similarity_func=box_iou_similarity,
            )
            instance_mask = instance_mask & sel

        if output_type == "_raw":
            return dict(
                pred_scores=np.asarray(scores)[instance_mask],
                pred_locations=np.asarray(preds["locations"])[instance_mask] / scaling,
                pred_bboxes=(np.asarray(bboxes) / np.r_[scaling, scaling])[instance_mask],
                pred_segmentations=np.asarray(preds["segmentations"])[instance_mask],
                pred_segmentation_pos=np.stack([
                    preds["segmentation_z0_coord"],
                    preds["segmentation_y0_coord"],
                    preds["segmentation_x0_coord"],
                ], axis=-1)[instance_mask],
            )
        elif output_type == "bbox":
            target_shape = np.broadcast_to([36], [len(orig_shape)])
            segmentations = crop_and_resize_patches(
                preds, bboxes,
                target_shape=target_shape,
                convert_logits=True,
            )
            results = dict(
                pred_scores=np.asarray(scores)[instance_mask],
                pred_bboxes=(np.asarray(bboxes) / np.r_[scaling, scaling])[instance_mask],
                pred_masks=np.asarray(segmentations)[instance_mask],
                pred_locations=np.asarray(preds["locations"])[instance_mask] / scaling,
            )

        elif output_type == "contour":
            if not is_3d:
                polygons, instance_mask = _to_polygons(
                    preds, instance_mask, segmentation_threshold=seg_logit_threshold,
                )
                polygons = [(c + 0.5) / scaling[::-1] for c in polygons]

                results = dict(
                    pred_scores=np.asarray(scores)[instance_mask],
                    pred_bboxes=(np.asarray(bboxes) / np.r_[scaling, scaling])[instance_mask],
                    pred_contours=polygons,
                )
            else:
                meshes, instance_mask = _to_mesh(
                    preds, instance_mask,
                    segmentation_threshold=seg_logit_threshold,
                )

                for mesh in meshes:
                    mesh['verts'] /=  scaling

                results = dict(
                    pred_scores=np.asarray(scores)[instance_mask],
                    pred_bboxes=(np.asarray(bboxes) / np.r_[scaling, scaling])[instance_mask],
                    pred_contours=meshes,
                )                

        else:  # Label
            label = patches_to_label(
                preds, reshape_to,
                mask=instance_mask,
                score_threshold=0,
                threshold=seg_logit_threshold,
            )
            if not (scaling == 1.0).all():
                label = jax.image.resize(
                    label, orig_shape, "nearest",
                )

            results = dict(
                pred_scores=np.asarray(scores)[instance_mask],
                pred_label=np.asarray(label),
            )

        elapsed = (time.time() - start_time) * 1000
        logging.debug(f"done formating output in {elapsed:.2f} ms")

        return results

    def predict_on_large_image(
        self,
        image: ArrayLike,
        *,
        reshape_to: tuple[int,...]|None = None,
        gs: int|None = None,
        ss: int|None = None,
        nms_iou: float = 0.0,
        score_threshold: float = 0.5,
        segmentation_threshold: float = 0.5,
        min_area: float = 0,
        min_cells_per_patch: int = 0,
        output_type: str = "label",
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
            min_area: Minimum area of a valid prediction.
            score_threshold: Min score needed to be included in the output.
            segmentation_threshold: Threshold value for segmentation.
            nms_iou: IOU threshold value for non-max-supression durint post-processing

        Returns:
            - pred_scores: The prediction scores of each instance.
            - pred_bboxes: The bounding-boxes of detected instances in y0x0y1x1 format
            - pred_masks: A 3d array representing segmentation within bboxes
        """
        if not output_type in ("bbox", "label", "contour"):
            raise ValueError(
                f"output_type should be 'bbox'|'label'|'contour'. Got {output_type} instead."
            )

        if image.ndim == 2 or image.shape[-1] > 3:
            logging.warning("input seems to have no channel dim. Add one")
            image = image[..., None]

        orig_shape = image.shape[:-1]
        assert image.ndim == 3 or image.ndim == 4, f"ilegal image dim {image.shape}"

        if ss is None:
            ss = 1024 if image.ndim == 3 else 192
            gs = None
        if gs is None:
            gs = ss + 64 if image.ndim == 3 else 32

        if reshape_to is None: 
            reshape_to = np.array(orig_shape)
        else:
            reshape_to = np.broadcast_to(reshape_to, [len(orig_shape)])
        scaling = reshape_to / orig_shape
        
        image = jax.image.resize(image, tuple(reshape_to) + (image.shape[-1],), "linear")
        image = image - image.mean()
        image = image / (image.std() + 1e-8)

        grid_positions = [slice(0, max(d-(gs-ss), 1), ss) for d in reshape_to]
        grid_positions = np.moveaxis(np.mgrid[grid_positions], 0, -1)
        grid_positions = grid_positions.reshape(-1, len(orig_shape))

        preds = []
        for pos in grid_positions:
            slices = (slice(x, x + gs) for x in pos)
            patch = image.__getitem__(tuple(slices))

            logging.info(f"Processing grid {pos} with patch size {patch.shape}")

            pred = self.predict(
                patch,
                score_threshold=score_threshold,
                segmentation_threshold=segmentation_threshold,
                min_area=min_area,
                nms_iou=nms_iou,
                normalize=False,
                output_type="bbox",
            )
            pred = _remove_edge_instances(
                pred, pos, reshape_to,
                gs=gs,
            )

            if len(pred['pred_scores']) > min_cells_per_patch:
                preds.append(dict(
                    pred_scores = pred['pred_scores'],
                    pred_bboxes = pred['pred_bboxes'] + np.r_[pos, pos],
                    pred_masks = pred['pred_masks'],
                    # pred_locations = pred["pred_locations"] + pos
                ))

        preds = jax.tree_util.tree_map(lambda *x: np.concatenate(x), *preds)

        # nms
        logging.info(f"nms...")

        asort = np.argsort(preds["pred_scores"])[::-1]
        preds = jax.tree_util.tree_map(lambda x: x[asort], preds)
        preds = _nms(preds, nms_iou)

        # rescale
        preds["pred_bboxes"] /= np.r_[scaling, scaling]
        # preds["pred_locations"] /= scaling

        if output_type == "bbox":
            return preds

        if output_type == "contour":
            if len(orig_shape) == 2:
                polygons = []
                _, sy, sx = preds['pred_masks'].shape
                for box, seg in zip(preds['pred_bboxes'], preds['pred_masks']):
                    y0, x0, y1, x1 = box
                    polygon = _get_contour((seg >= 0.5).astype("uint8"))
                    polygon = polygon / [sx, sy] * [x1-x0, y1-y0] + [x0, y0]
                    polygons.append(polygon)
                preds['pred_contours'] = polygons

            else:
                meshes = []
                _, sz, sy, sx = preds['pred_masks'].shape
                for box, seg in zip(preds['pred_bboxes'], preds['pred_masks']):
                    z0, y0, x0, z1, y1, x1 = box
                    mesh = _get_mesh((seg >= 0.5).astype("uint8"))
                    assert mesh is not None
                    mesh['verts'] = mesh['verts'] / [sz, sy, sx] * [z1-z0, y1-y0, x1-x0] + [z0, y0, x0]
                    meshes.append(mesh)

                preds['pred_contours'] = meshes

            return preds

        else: # label
            from scipy.interpolate import interpn
            logging.info(f"Generating label...")
            label = np.zeros(orig_shape, dtype=int)
            bboxes = preds['pred_bboxes']
            masks = preds['pred_masks']
            dim = bboxes.shape[-1] // 2

            if dim == 2:
                _, sy, sx = masks.shape
                mask_grid = (np.arange(sy), np.arange(sx))
            else:
                _, sz, sy, sx = masks.shape
                mask_grid = (np.arange(sz), np.arange(sy), np.arange(sx))

            p0s = np.maximum(0, np.floor(bboxes[:, :dim]).astype(int))
            p1s = np.minimum(np.ceil(bboxes[:, dim:]).astype(int), label.shape)

            for label_idx, (p0, p1, mask) in enumerate(zip(p0s, p1s, masks)):
                slices = [slice(a,b) for a,b in zip(p0, p1)]
                coords = tuple(np.mgrid[slices])
                src_coords = (np.stack(coords, axis=-1) + 0.5 - p0) / (p1-p0) * mask.shape - 0.5
                values = interpn(mask_grid, mask, src_coords, bounds_error=False, fill_value=0)
                values = (values >= segmentation_threshold ) * (label_idx + 1)
                label[coords] = np.where(
                    label[coords] == 0,
                    values,
                    label[coords],
                )
            preds['pred_label'] = label

            return preds

    @property
    def module(self) -> nn.Module:
        return self.model[0]

    @property
    def params(self) -> dict:
        return self.model[1]

    @params.setter
    def params(self, new_params):
        self.model = self.module, new_params
