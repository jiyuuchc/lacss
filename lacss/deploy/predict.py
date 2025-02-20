""" 
Attributes:
    model_urls: URLs for build-in pretrain models. e.g model_urls["default"].
"""
from __future__ import annotations

import logging
import math
import time
import os

from functools import reduce, partial
from typing import Mapping, Sequence, Tuple, Any

import cv2
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ..ops import patches_to_label, bboxes_of_patches, crop_and_resize_patches
from ..utils import load_from_pretrained
from ..typing import Array, ArrayLike

Shape = Sequence[int]

logger = logging.getLogger(__name__)

def _remove_edge_instances(pred, pos, patch_sz, img_sz):
    bboxes = pred['bboxes']
    removal = np.zeros([bboxes.shape[0]], dtype=bool)

    dim = len(img_sz)

    removal |= ((bboxes[:, :dim] <= 0) & (pos > 0)).any(axis=-1)
    removal |= ((bboxes[:, dim:] >= patch_sz) & (pos + patch_sz < img_sz)).any(axis=-1)

    pred["segmentation_is_valid"] &= ~removal


def _to_polygons(
    predictions, mask,
    *,
    segmentation_threshold=0.0,
    chain_approx=cv2.CHAIN_APPROX_SIMPLE,
) -> Tuple[Sequence[Any], np.ndarray]:
    polygons = []

    y0s = np.asarray(predictions["segmentation_y0_coord"])
    x0s = np.asarray(predictions["segmentation_x0_coord"])
    segs = np.asarray(predictions["segmentations"] >= segmentation_threshold).astype("uint8")
    mask = np.array(mask) 

    segs = segs.squeeze(1)

    for k in range(segs.shape[0]):
        if mask[k]:
            c, _ = cv2.findContours(segs[k], cv2.RETR_EXTERNAL, chain_approx)
            max_len_element = reduce(
                lambda a, b: a if len(a) >= len(b) else b,
                c,
                np.zeros([0, 1, 2], dtype=int),
            )
            polygon = max_len_element.squeeze(1).astype(float)
            if len(polygon) > 0:
                polygons.append(polygon + [x0s[k], y0s[k]] + .5)
            else:
                mask[k] = False

    return polygons, mask


def _to_mesh(
    predictions, mask,
    *,
    segmentation_threshold=0.0,
    step_size=1,
):
    from skimage.measure import marching_cubes
    meshes = []

    z0s = np.asarray(predictions["segmentation_z0_coord"])
    y0s = np.asarray(predictions["segmentation_y0_coord"])
    x0s = np.asarray(predictions["segmentation_x0_coord"])
    segs = np.asarray(predictions["segmentations"])
    mask = np.array(mask) 

    for k in range(segs.shape[0]):
        if mask[k]:
            try:
                # verts, faces, norms, _ = marching_cubes(seg, allow_degenerate=False)
                verts, faces, _, _ = marching_cubes(segs[k], step_size=step_size, level=segmentation_threshold)
                verts += [z0s[k], y0s[k], x0s[k]]

                meshes.append(dict(verts=verts, faces=faces))

            except:
                mask[k] = False

    return meshes, mask  


def _clean_up_mesh(mesh, n=500, regularization=0.1):
    from vedo import Mesh
    
    vedo_mesh = (
        Mesh([mesh["verts"], mesh["faces"]])
        .decimate(n=n, regularization=regularization)
    )
    return dict(
        verts=vedo_mesh.vertices,
        faces=np.array(vedo_mesh.cells),
    )


def _nms(boxes, iou_threshold, selected, asort=None):
    from lacss.ops import box_iou_similarity

    if asort is not None:
        boxes = boxes[asort]
        selected = selected[asort]

    sm = np.array(box_iou_similarity(boxes, boxes))
    sm = np.triu(sm, k=1)

    assert len(selected) == boxes.shape[0]

    for k in range(len(selected)):
        if selected[k]:
            selected &= sm[k] < iou_threshold
    
    # inverse selected` to the original order
    if asort is not None:
        selected = selected[asort.argsort()]

    return selected


def _predict_partial(model, params, image):
    model = model.copy()
    model.segmentor = None
    model.segmentor_3d = None

    lpn_out, seg_features = [], []
    for p in params:
        model_out =  model.apply(dict(params=p), image)
        lpn_out.append( model_out['detector'] )
        seg_features.append(model_out['seg_features'])

    lpn_out = jax.tree_util.tree_map(
        lambda *x: jnp.stack(x), * lpn_out
    )
    lpn_out = dict(
        logits = lpn_out['logits'].mean(axis=0),
        regressions =  lpn_out['regressions'].mean(axis=0),
        ref_locs = lpn_out['ref_locs'][0],
    )
    lpn_out['pred_locs'] = lpn_out['ref_locs'] + lpn_out['regressions'] * model.detector.feature_scale

    return seg_features, lpn_out


@partial(jax.jit, static_argnums=0)
def _model_fn(model, params, image):
    from lacss.modules.lpn import generate_predictions

    seg_features, lpn_out = _predict_partial(model, params, image)

    predictions = generate_predictions(model.detector, lpn_out)

    assert len(seg_features) == len(params)

    seg_predictions = []
    if image.ndim == 3 and model.segmentor is not None:
        for x, p in zip(seg_features, params):
            seg_predictions.append( 
                model.segmentor.apply(
                    dict(params=p['segmentor']), 
                    x, 
                    predictions['locations']
                )['predictions'] 
            )

    elif image.ndim == 4 and model.segmentor_3d is not None:
        seg_predictions = []
        for x, p in zip(seg_features, params):
            seg_predictions.append( 
                model.segmentor_3d.apply(
                    dict(params=p['segmentor_3d']), 
                    x, 
                    predictions['locations']
                )['predictions'] 
            )

    if len(seg_predictions) > 0:
        predictions['segmentations'] = jnp.mean(
            jnp.array([ x['segmentations'] for x in seg_predictions]),
            axis = 0,
        )

        seg_prediction = seg_predictions[0]
        del seg_prediction['segmentations']

        predictions.update( seg_prediction )
        
    return predictions


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
            url: str | os.PathLike | tuple[nn.Module, dict],
            *,
            f16 = False,
            grid_size:int = 1088,
            step_size:int = 1024,
            grid_size_3d:int = 384,
            step_size_3d:int = 320,
            mask_size:int = 36,
            mc_step_size:int = 2,
        ):
        """Construct Predictor

        Args:
            url: A URL or local path to the saved model.
                URLs for build-in pretrained models can be found in lacss.deploy.model_urls
        """

        if isinstance(url, tuple) and len(url) == 2:
            if not isinstance(url[0], nn.Module):
                raise ValueError(
                    "Initiaize the Predictor with a tuple, but the first element is not a Module."
                )

            module, params = url

        else:
            module, params = load_from_pretrained(url)


        if isinstance(params, Mapping):
            params = [params]

        self.module = module
        self.params = params

        self.f16 = f16
        if self.f16:
            self.params = jax.tree_util.tree_map(
                lambda x: x.astype("float16"), self.params,
            )

        assert step_size < grid_size, f"step_size ({step_size}) not smaller than grid_size ({grid_size})"
        assert grid_size % 32 == 0, f"grid_size ({grid_size}) is not divisable by 32"
        assert step_size_3d < grid_size_3d, f"step_size ({step_size_3d}) not smaller than grid_size ({grid_size_3d})"
        assert grid_size_3d % 32 == 0, f"grid_size ({grid_size_3d}) is not divisable by 32"

        self.gs = grid_size
        self.ss = step_size

        self.gs_3d = grid_size_3d
        self.ss_3d = step_size_3d

        self.mask_size = mask_size
        self.mc_step_size = mc_step_size


    def _format_image(self, image, target_shape, normalize=True):
        from skimage.transform import resize

        # process the data in float32 to avoid overflow
        # only convert to float16 (if needed) at the end

        image = image.astype("float32")

        orig_shape = image.shape[:-1]        
        
        if normalize:
            image = image - image.mean()
            image = image / (image.std() + 1e-6)

        if target_shape is None: 
            target_shape = np.array(orig_shape)

        else:
            target_shape = np.broadcast_to(target_shape, [len(orig_shape)])

            scaling = target_shape / orig_shape

            if np.all(np.abs(scaling - 1) < 0.1): #ignore small scaling
                target_shape = np.array(orig_shape)

            else:
                image = resize(image, target_shape)

        if self.f16:
            image = image.astype("float16")

        logger.debug(f"resized image data from {orig_shape} to {target_shape}")

        return image, target_shape


    def _call_model(self, image, threshold, to_cpu=True):
        logger.debug(f"process image patch of {image.shape}")

        # pad image to fixed input sizes
        img_sz = image.shape[:-1]
        padding_shape = [(x - 1) // 32 * 32 + 32 for x in img_sz]

        if image.ndim == 3: #2D input
            if max(padding_shape) <=  544:
                padding_shape = [544, 544]

            elif max(padding_shape) <=  1088:
                padding_shape = [1088, 1088]

        else:
            # for 3D we require isotropic dimension
            padding_shape = [max(padding_shape)] * 3

            if max(padding_shape) <= 256:
                padding_shape = [256, 256, 256]

            elif max(padding_shape) <=  384:
                padding_shape = [384, 384, 384]

        padding = [ [0, s - s0] for s, s0 in zip(padding_shape, image.shape[:-1])]
        padding += [[0, 0]]

        image = np.pad(image, padding)

        # make 3-ch
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 2:
            image = np.stack([image, np.zeros_like(image[..., :1])], axis=-1)

        logger.debug(f"call model with data {image.shape}")
        predictions = _model_fn(self.module, self.params, image)

        predictions["bboxes"] = bboxes_of_patches(
            predictions,
            threshold=threshold,
            image_shape=img_sz,
        )

        if to_cpu:
            predictions = jax.tree_util.tree_map(
                lambda x: np.array(x),
                predictions,
            )

        return predictions


    def _process_image(self, image, threshold):
        img_sz = image.shape[:-1]
        img_dim = len(img_sz)

        # no grid for small images
        if (
            (img_dim == 2 and max(img_sz) <= self.gs) or
            (img_dim == 3 and max(img_sz) <= self.gs_3d)
         ) :
            return self._call_model(image, threshold), True

        
        gs, ss = (self.gs, self.ss) if img_dim == 2 else (self.gs_3d, self.ss_3d)

        grid_positions = [slice(0, max(d-(gs-ss), 1), ss) for d in img_sz]
        grid_positions = np.moveaxis(np.mgrid[grid_positions], 0, -1)
        grid_positions = grid_positions.reshape(-1, img_dim)

        predictions = []
        for pos in grid_positions:
            slices = (slice(x, x + gs) for x in pos)
            patch = image.__getitem__(tuple(slices))
            patch_sz = patch.shape[:-1]
            
            patch_pred = self._call_model(patch, threshold)
            _remove_edge_instances(patch_pred, pos, patch_sz, img_sz)

            n_det = np.count_nonzero(patch_pred["segmentation_is_valid"])
            if n_det > 0:
                z0, y0, x0 = pos if img_dim == 3 else (0, pos[0], pos[1])
                patch_pred["bboxes"] += np.r_[pos, pos]
                patch_pred["segmentation_z0_coord"] += z0
                patch_pred["segmentation_y0_coord"] += y0
                patch_pred["segmentation_x0_coord"] += x0
                predictions.append(patch_pred)
            
        preds = jax.tree_util.tree_map(lambda *x: np.concatenate(x), *predictions)

        return preds, False


    def _filter_predictions(self, predictions, score_threshold, segmentation_threshold, min_area, nms_iou, is_sorted):
        instance_mask = predictions["segmentation_is_valid"]

        instance_mask &= predictions["scores"] >= score_threshold

        if min_area > 0:
            areas = np.count_nonzero(
                predictions["segmentations"] > segmentation_threshold, 
                axis=(1, 2, 3),
            )
    
            instance_mask &= areas >= min_area

        if nms_iou > 0 and nms_iou < 1:
            instance_mask = _nms(
                predictions["bboxes"], 
                nms_iou,
                instance_mask,
                predictions["scores"].argsort()[::-1] if not is_sorted else None,
            )

        return instance_mask


    def predict(
        self,
        image: ArrayLike,
        *,
        output_type: str = "label",
        reshape_to: int|tuple[int]|np.ndarray|None = None,
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

        is_3d = image.ndim == 4

        image, reshape_to = self._format_image(image, reshape_to, normalize=normalize)

        scaling = reshape_to / img_shape

        seg_logit_threshold = math.log(segmentation_threshold / (1 - segmentation_threshold))

        logger.debug(f"done preprocessing")

        preds, is_sorted = self._process_image(image, seg_logit_threshold)

        logger.debug(f"filter detections...")
        instance_mask = self._filter_predictions(
            predictions=preds,
            score_threshold=score_threshold, 
            segmentation_threshold=segmentation_threshold,
            min_area=min_area * np.prod(scaling),
            nms_iou=nms_iou,
            is_sorted=is_sorted,
        )
        logger.debug(f"filter detections...done")

        # generate outputs
        if output_type == "_raw":
            preds["segmentation_is_valid"] = instance_mask
            return preds

        elif output_type == "bbox":
            target_shape = np.broadcast_to([self.mask_size], [len(img_shape)])
            segmentations = crop_and_resize_patches(
                preds, preds["bboxes"],
                target_shape=tuple(target_shape),
                convert_logits=True,
            )
            results = dict(
                pred_scores=preds["scores"][instance_mask],
                pred_bboxes=preds["bboxes"][instance_mask] / np.r_[scaling, scaling],
                pred_masks=np.asarray(segmentations)[instance_mask],
                pred_locations=preds["locations"][instance_mask] / scaling,
            )

        elif output_type == "contour":
            if not is_3d:
                polygons, instance_mask = _to_polygons(
                    preds, instance_mask, 
                    segmentation_threshold=seg_logit_threshold,
                )
                polygons = [c / scaling[::-1] for c in polygons]

                results = dict(
                    pred_scores=preds["scores"][instance_mask],
                    pred_bboxes=preds["bboxes"][instance_mask] / np.r_[scaling, scaling],
                    pred_contours=polygons,
                )
            else:
                meshes, instance_mask = _to_mesh(
                    preds, instance_mask,
                    segmentation_threshold=seg_logit_threshold,
                    step_size=self.mc_step_size,
                )

                for mesh in meshes:
                    mesh['verts'] /=  scaling

                # clean up
                meshes = [_clean_up_mesh(m) for m in meshes]

                results = dict(
                    pred_scores=preds["scores"][instance_mask],
                    pred_bboxes=preds["bboxes"][instance_mask] / np.r_[scaling, scaling],
                    pred_contours=meshes,
                )                

        else:  # Label
            label = patches_to_label(
                preds, 
                tuple(reshape_to),
                mask=instance_mask,
                score_threshold=0,
                threshold=seg_logit_threshold,
            )
            if not (scaling == 1.0).all():
                label = jax.image.resize(
                    label, img_shape, "nearest",
                )

            results = dict(
                pred_scores=preds["scores"][instance_mask],
                pred_label=np.asarray(label),
            )

        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"done prediction in {elapsed:.2f} ms")

        return results

