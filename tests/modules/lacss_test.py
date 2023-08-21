import pytest

from lacss.modules import Lacss


def test_module_init():
    m = Lacss()
    cfg = m.get_config()
    m2 = Lacss.from_config(cfg)

    assert isinstance(m2, Lacss)
    assert m2.get_config() == cfg
    assert m2 == m


def test_module_cfg():
    cfg = {}
    m = Lacss.from_config(cfg)
    assert m == Lacss()
