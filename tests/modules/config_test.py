import pytest

def test_config():
    from lacss.modules import Lacss
    model = Lacss.get_default_model()
    cfg = model.get_config()
    model2 = Lacss.from_config(cfg)

    assert model == model2
