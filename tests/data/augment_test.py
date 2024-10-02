import pytest
import numpy as np

from lacss.data.augment_ import *

@pytest.fixture
def test_data_2d():
    np.random.seed(1234)
    imgsize = [64, 50]
    image = np.random.random(imgsize + [3])
    centroids = np.random.random([10, 2]) * imgsize
    sizes = np.random.random([10, 2]) * 10
    bboxes = np.c_[centroids - sizes/2, centroids + sizes/2]
    bboxes = np.clip(bboxes, 0, np.r_[imgsize, imgsize])
    return dict(
        image = image,
        image_mask = (image > 0.3).all(axis=-1),
        centroids = centroids,
        bboxes = bboxes,
    )

@pytest.fixture
def test_data_3d():
    np.random.seed(1234)
    imgsize = [32, 64, 50]
    image = np.random.random(imgsize + [3])
    centroids = np.random.random([10, 3]) * imgsize
    sizes = np.random.random([10, 3]) * 10
    bboxes = np.c_[centroids - sizes/2, centroids + sizes/2]
    bboxes = np.clip(bboxes, 0, np.r_[imgsize, imgsize])
    return dict(
        image = image,
        image_mask = (image > 0.3).all(axis=-1),
        centroids = centroids,
        bboxes = bboxes,
    )

def assert_same_data(a, b):
    assert np.allclose(a['image'], b['image'])
    assert np.allclose(a['bboxes'], b['bboxes'])
    assert np.allclose(a['centroids'], b['centroids'])

def test_flip_image(test_data_2d):
    orig = test_data_2d

    flipped = flip_left_right(flip_left_right(orig))
    assert_same_data(orig, flipped)

    flipped = flip_up_down(flip_left_right(orig))
    assert_same_data(orig, flipped)

def test_flip_image_3d(test_data_3d):
    orig = test_data_3d

    flipped = flip_left_right(flip_left_right(orig))
    assert_same_data(orig, flipped)

    flipped = flip_up_down(flip_left_right(orig))
    assert_same_data(orig, flipped)

    flipped = flip_top_bottom(flip_left_right(orig))
    assert_same_data(orig, flipped)


def test_crop(test_data_2d):
    orig = test_data_2d

    roi=[5, 6, 20, 30]
    expected_size = (roi[2]-roi[0], roi[3]-roi[1])
    cropped = crop_to_roi(orig, roi=roi)
    assert cropped['image'].shape[:-1] == expected_size
    assert cropped['image_mask'].shape == expected_size

