import pytest
import tensorflow as tf

import lacss.train


def test_tf_adaptor():
    ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    ds_it = iter(lacss.train.TFDatasetAdapter(ds))

    assert next(ds_it) == 1
    assert next(ds_it) == 2
