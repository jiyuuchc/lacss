from __future__ import annotations

from pathlib import Path
import ml_collections

DATAPATH = Path("/home/FCAM/jyu/datasets")

def get_config():
    import livecell_dataset
    from lacss.modules import Lacss

    config = ml_collections.ConfigDict()
    config.name = "lacss_livecell_test"

    config.train = ml_collections.ConfigDict()
    config.train.seed = 42
    config.train.batchsize = 3
    config.train.steps = 100000
    config.train.validation_interval = 10000
    config.train.lr = 5e-5
    config.train.weight_decay = 1e-3

    config.model = Lacss.get_preconfigued("small")

    livecell_train, livecell_val = livecell_dataset.get_data(DATAPATH)
    config.data = ml_collections.ConfigDict()
    config.data.ds_train = (
        livecell_train
        .repeat()
        .batch(config.train.batchsize)
        .prefetch(1)
    )
    config.data.ds_val = livecell_val.prefetch(1)

    return config
