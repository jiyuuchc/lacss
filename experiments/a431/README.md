### A431 models

#### Dataset
Download [here](https://drive.google.com/drive/folders/15aFQsONxHt_9PIgQiri_wxqqPchFJ5Oq).

Unzip to a new directory

#### Transfer learning
Download the pretrained [lacss model](https://drive.google.com/drive/folders/1zuUDOpqSNrN7C4oHwfHT8qHnFrruTn6_)

This model was pretrained on LIVECell dataset under full supervision

#### Training

    CONFIG_FILE=<project dir>/lacss/experiments/configs/A431/config.json
    python <project dir>/lacss/experiments/a431/train_val.py <log_dir> --config $CONFIG_FILE --transfer <path_to_pretrained_model>
