### LIVECell models

#### Download dataset

The [LIVECell dataset](https://sartorius-research.github.io/LIVECell/). You will need an aws account.

#### Train Fully-supervised:

```
python <project_dir>/experiments/livecell/supervised.py <model_config_file> --datapath=<data_dir> --logpath=<log_dir>
```

The model_config_file is a json file defining model hyperparameters. Example configs are in directory experiments/configs.

#### Train Point-supversied

```
python <project_dir>/experiments/livecell/semisupervised.py <model_config_file> --datapath=<data_dir> --logpath=<log_dir> 
```

If starting from an existing transfer model:
```
python <project_dir>/experiments/livecell/semisupervised.py <model_config_file> <path-to-transfer-model> --datapath=<data_dir> --logpath=<log_dir> --warmup-steps=0
```

#### Testing: 

```
python <project_dir>/experiments/livecell/eval.py <path to saved_model or training checkpoint>
```

