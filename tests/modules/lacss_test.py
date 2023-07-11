from functools import partial

import flax.linen as nn
import jax
import optax
import pytest

from lacss.modules import Lacss

cfg_json = """
{
  "backbone_config": {
  	"drop_path_rate": 0.4,
    "out_channels": 256
  },
  "lpn_config": {
    "conv_spec": [[256, 256, 256, 256], []]
  },
  "detector_config": {
  	"train_max_output": 2560,
  	"test_max_output": 2560
  },
  "segmentor_config": {
    "conv_spec": [[256, 256, 256], [64]]
  }  
}
"""


def test_module_init():
    import json

    cfg_dict = json.loads(cfg_json)
    m = Lacss(**cfg_dict)

    assert isinstance(m, Lacss)

    new_cfg_dict = m.get_config()
    m2 = Lacss(**new_cfg_dict)

    assert isinstance(m2, Lacss)

    assert m == m2
