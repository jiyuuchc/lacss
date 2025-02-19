import pytest

from pathlib import Path

import jax
import numpy as np

from lacss.deploy.predict import Predictor
from lacss.modules import Lacss

jnp = jax.numpy

MODULE_DIR = Path(__file__).parent

@pytest.fixture(scope='module')
def model():
    from lacss.deploy.predict import Predictor

    cache_dir = Path.home() / ".cache" / "lacss"
    model_file = cache_dir / "lacss3_small"

    if not model_file.exists():
        import urllib.request
    
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        urllib.request.urlretrieve(
            "https://huggingface.co/jiyuuchc/lacss3-small/resolve/main/lacss3-small?download=true",
            model_file,
        )

    predictor = Predictor(model_file)

    yield predictor

    del predictor


@pytest.fixture(scope='module')
def test_image_2d():
    import tifffile
    img = tifffile.imread(MODULE_DIR / "test_data" / "test_2d.tif")

    yield img

    del img


@pytest.fixture(scope='module')
def test_image_3d():
    import tifffile
    img = tifffile.imread(MODULE_DIR / "test_data" / "test_3d.tif")

    yield img

    del img


def test_predict_2d(model, test_image_2d):
    label = model.predict(test_image_2d)["pred_label"]

    assert label.shape == test_image_2d.shape

    assert label.max() == 22

    label = model.predict(test_image_2d, nms_iou=0.4)["pred_label"]

    assert label.max() == 22

    preds = model.predict(test_image_2d, output_type="contour")

    assert len(preds['pred_scores']) == 22
    assert len(preds['pred_contours']) == 22


def test_predict_3d(model, test_image_3d):
    preds = model.predict(
        test_image_3d, 
        score_threshold=0.3,
        reshape_to=(60, 256, 256),
        nms_iou=0.4,
        output_type="contour",
    )

    assert len(preds['pred_scores']) > 0
    assert len(preds['pred_scores']) == len(preds['pred_contours'])
