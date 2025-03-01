import numpy as np
import pytest


def test_grid_predict(model, test_image_2d):
    big_img = np.hstack([test_image_2d, test_image_2d])

    pred = model.predict(big_img, nms_iou=0.4)

    label = pred["pred_label"]

    assert label.shape == big_img.shape

    assert label.max() > 0
    assert label.max() == pred["pred_scores"].shape[0]


def test_grid_predict_3d(model, test_image_3d):
    big_img = np.hstack([test_image_3d, test_image_3d])

    pred = model.predict(
        big_img,
        score_threshold=0.3,
        nms_iou=0.4,
    )
    label = pred["pred_label"]

    assert label.shape == big_img.shape

    assert label.max() > 0
    assert label.max() == pred["pred_scores"].shape[0]
