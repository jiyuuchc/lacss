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
import optax
from flax.core.frozen_dict import freeze, unfreeze
from flax.training.train_state import TrainState

import lacss.modules
import lacss.train.strategy as strategy
from lacss.ops import patches_to_label

_cached_partial = lru_cache(partial)


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
        import urllib

        if os.path.isfile(pretrained):
            pretrained = "file://" + urllib.request.pathname2url(
                os.path.abspath(pretrained)
            )

        bytes = urllib.request.urlopen(pretrained).read()
        thingy = pickle.loads(bytes)

        if isinstance(thingy, lacss.train.Trainer):

            module = thingy.model
            params = thingy.params

            if not isinstance(module, lacss.modules.Lacss):

                module = module.bind(dict(params=params))
                module, params = module._lacss.unbind()
                params = params["params"]

            cfg = module

        else:

            cfg, params = thingy

    if "params" in params and len(params) == 1:

        params = params["params"]

    # making a round-trip of module->cfg->module to icnrease backward-compatibility
    if isinstance(cfg, nn.Module):

        cfg = dataclasses.asdict(cfg)

    module = lacss.modules.Lacss(**cfg)

    return module, freeze(params)


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

    def predict(self, image, **kwargs):

        module, params = self.model

        if len(kwargs) > 0:
            apply_fn = _cached_partial(module.apply, **kwargs)
        else:
            apply_fn = module.apply

        return _predict(apply_fn, params, image)

    def predict_label(self, image, **kwargs):

        preds = self.predict(image, **kwargs)

        return patches_to_label(preds, input_size=image.shape[:2])
