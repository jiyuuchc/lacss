""" 
Attributes:
    model_urls: URLs for build-in pretrain models. e.g model_urls["livecell"].
"""

import dataclasses
import json
import logging
import os
import pickle
import typing as tp
from functools import lru_cache, partial
from logging.config import valid_ident
from os.path import join

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.training.train_state import TrainState

import lacss.modules
import lacss.train.strategy as strategy
from lacss.ops import patches_to_label
from lacss.train import Inputs

_Image = jnp.ndarray
_ImageShape = tp.Tuple[int, int, int]
_Params = FrozenDict

_cached_partial = lru_cache(partial)

model_urls: tp.Mapping[str, str] = {
    "livecell": "https://data.mendeley.com/public-files/datasets/sj3vrvm6w3/files/439e524f-e4e9-4f97-9f38-c22cb85adbd1/file_downloaded",
    "tissuenet": "https://data.mendeley.com/public-files/datasets/sj3vrvm6w3/files/1e0a839d-f564-4ee0-a4f3-34792df7c613/file_downloaded",
}


def load_from_pretrained(pretrained):
    if os.path.isdir(pretrained):
        import json

        try:
            with open(join(pretrained, "config.in")) as f:
                cfg = json.load(f)
        except:
            raise ValueError(
                f"Cannot open 'config.in' in {pretrained} as a json config file."
            )
        with open(join(pretrained, "weights.pkl")) as f:
            params = pickle.load(f)

    else:
        if os.path.isfile(pretrained):
            with open(pretrained, "rb") as f:
                thingy = pickle.load(f)

        else:
            import io
            from urllib.request import Request, urlopen

            headers = {"User-Agent": "Wget/1.13.4 (linux-gnu)"}
            req = Request(url=pretrained, headers=headers)

            bytes = urlopen(req).read()
            thingy = pickle.loads(bytes)

        if isinstance(thingy, lacss.train.Trainer):
            module = thingy.model
            params = thingy.params

            if not isinstance(module, lacss.modules.Lacss):
                module = module.bind(dict(params=params))
                module, params = module.lacss.unbind()
                params = params["params"]

            cfg = module

        else:
            cfg, params = thingy

    if "params" in params and len(params) == 1:
        params = params["params"]

    # making a round-trip of module->cfg->module to icnrease backward-compatibility
    # if isinstance(cfg, nn.Module):

    #     cfg = dataclasses.asdict(cfg)

    # module = lacss.modules.Lacss(**cfg)

    return cfg, freeze(params)


@partial(jax.jit, static_argnums=0)
def _predict(apply_fn, params, image):
    preds = apply_fn(dict(params=params), image)
    preds["pred_bboxes"] = lacss.ops.bboxes_of_patches(preds)

    del preds["encoder_features"]
    del preds["decoder_features"]
    del preds["lpn_features"]
    del preds["lpn_scores"]
    del preds["lpn_regressions"]

    return preds


class Predictor:
    """Main class interface for model deployment.

    Attributes:
        module: The underlying FLAX module
        params: Model weights.
        detector: The detector submodule for convinence. A common use case
            is to customize this submodule during inference. e.g.
            ```
            predictor.detector['test_min_score"] = 0.5
            ```
            Detector submodule has no trained paramters.
    Args:

        url: A URL or local path to the saved model.
            URLs for build-in pretrained models can be found in lacss.deploy.model_urls
        precompile_shape: Image shape(s) for precompiling. Otherwise
            the model will be recompiled for every new input image shape.

    Examples:
        A typical use is to retrieve a build-in pretrained model:

            url = lacss.deploy.model_urls["livecell"]
            predictor = lacss.deploy.Predictor(url)
            label = predictor.predict_label(image)

    """

    def __init__(self, url: str, precompile_shape: tp.Optional[_ImageShape] = None):
        self.model = load_from_pretrained(url)

        if precompile_shape is not None:
            logging.info("Prcompile the predictor for image shape {precompile_shape}.")
            logging.info("This will take several minutes.")

            try:
                iter(precompile_shape[0])
            except:
                precompile_shape = [precompile_shape]
            for shape in precompile_shape:
                x = np.zeros(shape)
                _ = self.predict(x)

    def __call__(self, inputs, **kwargs):
        inputs_obj = Inputs.from_value(inputs)

        return self.predict(*inputs_obj.args, **inputs_obj.kwargs, **kwargs)

    def predict(
        self,
        image: _Image,
        min_area: float = 0,
        remove_out_of_bound: bool = False,
        scaling: float = 1.0,
        **kwargs,
    ) -> dict:
        """Predict segmentation.

        Args:
            image: A ndarray of (h,w,c) format. Value of c must be 1-3
            min_area: Minimum area of a valid prediction.
            remove_out_of_bound: Whether to remove out-of-bound predictions. Default is False.
            scaling: A image scaling factor. If not 1, the input image will be resized internally before fed
                to the model. The results will be resized back to the scale of the orginal input image.

        Returns:
            Model predictions with following elements:

                - pred_scores: The prediction scores of each instance.
                - pred_bboxes: The bounding-boxes of detected instances in y0x0y1x1 format
                - instance_output: A 3d array representing segmentation instances.
                - instance_yc: The meshgrid y-coordinates of the instances.
                - instance_xc: The meshgrid x-coordinates of the instances.
                - instance_mask: A boolean mask indicating the validity of the instances.
        """

        module, params = self.model

        if len(kwargs) > 0:
            apply_fn = _cached_partial(module.apply, **kwargs)
        else:
            apply_fn = module.apply

        # scale image if asked
        if scaling != 1.0:
            h, w, c = image.shape
            image = jax.image.resize(
                image, [round(h * scaling), round(w * scaling), c], "linear"
            )

        preds = _predict(apply_fn, params, image)

        # check prediction validity
        instance_mask = preds["instance_mask"].squeeze(axis=(1, 2))

        if min_area > 0:
            areas = jnp.count_nonzero(preds["instance_logit"] > 0, axis=(1, 2))
            instance_mask &= areas > min_area

        if remove_out_of_bound:
            pred_locations = preds["pred_locations"]
            h, w, _ = image.shape
            valid_locs = (pred_locations >= 0).all(axis=-1)
            valid_locs &= pred_locations[:, 0] < h
            valid_locs &= pred_locations[:, 1] < w
            instance_mask &= valid_locs

        preds["instance_mask"] = instance_mask.reshape(-1, 1, 1)

        # rescale segmentations
        if scaling != 1.0:
            (
                preds["instance_output"],
                preds["instance_yc"],
                preds["instance_xc"],
            ) = lacss.ops.rescale_patches(preds, 1 / scaling)
            preds["pred_locations"] = preds["pred_locations"] / scaling
            preds["pred_bboxes"] = preds["pred_bboxes"] / scaling

        return preds

    def predict_label(
        self, image: _Image, *, score_threshold: float = 0.5, **kwargs
    ) -> jnp.ndarray:
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

    @property
    def module(self) -> nn.Module:
        return self.model[0]

    @property
    def params(self) -> _Params:
        return self.model[1]

    @params.setter
    def params(self, new_params):
        self.model = self.module, new_params

    @property
    def detector(self) -> lacss.module.Detector:
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
