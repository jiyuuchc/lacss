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
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState

import lacss.modules
import lacss.train.strategy as strategy
from lacss.ops import patches_to_label
from lacss.train import Inputs

_cached_partial = lru_cache(partial)

Image = tp.Any

model_urls = {
    "livecell": "https://data.mendeley.com/public-files/datasets/sj3vrvm6w3/files/439e524f-e4e9-4f97-9f38-c22cb85adbd1/file_downloaded",
    "tissuenet": "https://data.mendeley.com/public-files/datasets/sj3vrvm6w3/files/1e0a839d-f564-4ee0-a4f3-34792df7c613/file_downloaded"
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
            from urllib.request import Request, urlopen
            import io
    
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
    return apply_fn(dict(params=params), image)


class Predictor:
    def __init__(self, url, precompile_shape=None):
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
        image: Image,
        min_area: float = 0,
        remove_out_of_bound: bool = False,
        **kwargs,
    ):
        """Predict segmentation.

        Args:
            image: ndarray of (h,w,c) format. Value of c must be 1-3
            min_area: Minimum area of a valid prediction. Default is 0
            remove_out_of_bound: Whether to remove out-of-bound predictions. Default is False.

        Returns:
            Raw model prediction as a dict. Most important keys are

            instance_output: Segmentation instances. (N, S, S) array
            instance_yc: The meshgrid y-coordinates of the instances. (N, S, S) array
            instance_xc: The meshgrid x-coordinates of the instances. (N, S, S) array
            pred_scores: The prediction scores of each instance. (N) array. -1 indicating invalid predictions.
            pred_locations: The prediction centroids of each instances. (N, 2) array
        """

        module, params = self.model

        if len(kwargs) > 0:
            apply_fn = _cached_partial(module.apply, **kwargs)
        else:
            apply_fn = module.apply

        preds = _predict(apply_fn, params, image)

        if min_area > 0:
            areas = jnp.count_nonzero(preds["instance_logit"] > 0, axis=(1, 2))
            preds["pred_scores"] = jnp.where(
                areas > min_area,
                preds["pred_scores"],
                -1,
            )

        if remove_out_of_bound:
            pred_locations = preds["pred_locations"]
            h, w, _ = image.shape
            valid_locs = (pred_locations >= 0).all(axis=-1)
            valid_locs &= pred_locations[:, 0] < h
            valid_locs &= pred_locations[:, 1] < w
            preds["pred_locations"] = jnp.where(
                valid_locs[:, None],
                preds["pred_locations"],
                -1,
            )
            preds["pred_scores"] = jnp.where(
                valid_locs,
                preds["pred_scores"],
                -1,
            )

        return preds

    def predict_label(self, image, **kwargs):
        preds = self.predict(image, **kwargs)

        return patches_to_label(preds, input_size=image.shape[:2])

    @property
    def module(self):
        return self.model[0]

    @property
    def detector(self):
        return self.module.detector

    #FIXME Change the detector without first compile the mode will result
    #in error. This is a hidden gotcha. Need to setup warning to the user.
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
