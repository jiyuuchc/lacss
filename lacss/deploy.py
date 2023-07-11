""" 
Attributes:
    model_urls: URLs for build-in pretrain models. e.g model_urls["livecell"].
"""
from __future__ import annotations

import logging
import os
import pickle
from functools import lru_cache, partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze

import lacss.modules
from lacss.ops import patches_to_label
from lacss.train import Inputs
from lacss.types import *

_cached_partial = lru_cache(partial)

model_urls: Mapping[str, str] = {
    "livecell": "https://data.mendeley.com/public-files/datasets/sj3vrvm6w3/files/439e524f-e4e9-4f97-9f38-c22cb85adbd1/file_downloaded",
    "tissuenet": "https://data.mendeley.com/public-files/datasets/sj3vrvm6w3/files/1e0a839d-f564-4ee0-a4f3-34792df7c613/file_downloaded",
}


def load_from_pretrained(pretrained):
    if os.path.isdir(pretrained):
        from lacss.train import LacssTrainer

        trainer = LacssTrainer.from_checkpoint(pretrained)
        module = trainer.model.principal
        params = trainer.params["principal"]

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

        else:
            cfg, params = thingy

            if isinstance(cfg, lacss.modules.Lacss):
                module = cfg
            else:
                module = lacss.modules.Lacss(**cfg)

    if "params" in params and len(params) == 1:
        params = params["params"]

    return module, freeze(params)


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

            ```
            import lacss.deploy

            # look up the url of a build-in mode
            url = lacss.deploy.model_urls["livecell"]

            # create the predictor instance
            predictor = lacss.deploy.Predictor(url)

            # make a prediction
            label = predictor.predict_label(image)

            ```
    Attributes:
        module: The underlying FLAX module
        params: Model weights.
        detector: The detector submodule for convinence. A common use case
            is to customize this submodule during inference. e.g.
            ```
            predictor.detector['test_min_score"] = 0.5
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

    def __call__(self, inputs, **kwargs):
        inputs_obj = Inputs.from_value(inputs)

        return self.predict(*inputs_obj.args, **inputs_obj.kwargs, **kwargs)

    def predict(
        self,
        image: ArrayLike,
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
                - instance_mask: a masking tensor. Invalid instances are maked with 0.
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
            areas = np.count_nonzero(preds["instance_logit"] > 0, axis=(1, 2))
            instance_mask &= areas > min_area

        if remove_out_of_bound:
            pred_locations = preds["pred_locations"]
            h, w, _ = image.shape
            valid_locs = (pred_locations >= 0).all(axis=-1)
            valid_locs &= pred_locations[:, 0] < h
            valid_locs &= pred_locations[:, 1] < w
            instance_mask &= valid_locs

        preds["instance_mask"] = instance_mask.reshape(-1, 1, 1)

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

    def pickle(self, save_path) -> None:
        """Save the model by pickling.
            In the form of (model_config:dict, weights:FrozenDict).

        Args:
            save_path: Path to the pkl file
        """
        _cfg = self.model[0].get_config()
        _params = self.model[1]

        with open(save_path, "wb") as f:
            pickle.dump((_cfg, _params), f)
