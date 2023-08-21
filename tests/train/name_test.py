from __future__ import annotations

import pytest

from lacss.train.utils import _get_name


def my_test_func(x):
    pass


def test_partial_func_name():
    from functools import partial

    _f = partial(my_test_func, x=1)

    assert _get_name(_f) == "my_test_func"

    def inner_func(x):
        pass

    _f2 = partial(inner_func, x=1)
    assert _get_name(_f2) == "inner_func"
