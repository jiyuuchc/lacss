### LIVECell models

#### Download dataset

The [LIVECell dataset](https://sartorius-research.github.io/LIVECell/). You will need an aws account.

#### TFRecord

Convert the original dataset into TRFRecord format.

    cd lacss/experiments/livecell
    python
    >>> from data import create_tfrecord
    >>> data_dir = ...
    >>> create_tfrecord(data_dir, extract_zip=True)

This creates three TFRecord files (train, val and test).

#### Train:

    export PYTHONPATH=<project_dir>:$PYTHONPATH
    python <project_dir>/experiments/livecell/train.py <data_dir> <log_dir> --config <model_config_file> --celltype -1|[0-7]

The model_config_file is a json file defining model hyperparameters. Recommended configs are in directory experiments/configs

For semi-supervised training, we use different model_config for different cell lines. The accuracy were particularily sensitive to lpn_level. For fully-supervised model, we typically train with celltype=-1 (all cell lines).

The losses and validation metrics (only maskAP50) were logged in tensorboard format in the log_dir. Model checkpoints were saved in the same place.

Interrupted training will resume by running the same command.

#### Testing: 
computing maskAP50-95 and FNRs on the testing set.

    python <project_dir>/experiments/livecell/test.py <data_dir> <log_dir> --checkpoint <checkpoint name>


