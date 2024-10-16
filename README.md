# LACSS
LACSS is a deep-learning model for 2D/3D single-cell segmentation from microscopy images.
```sh
   pip install lacss
```

## Models checkpoints
#### Multi-modality (2D + 3D)
| name | #params | download | mAP LiveCell* | mAP Cellpose* | mAP NIPS* | ovule (3D)* | platynereis (3D)* |
| :--- | --- | --- | :---: | :---: | :---: | :---: | :---: | 
| small | 60M | [model](https://huggingface.co/jiyuuchc/lacss3-small/resolve/main/lacss3-small)| 56.3 | 52.0 | 54.2 | 44.4 | 56.7 |
| base | 152M | [model](https://huggingface.co/jiyuuchc/lacss3-base/resolve/main/lacss3-base)| 57.1 | 56.0 | 62.9 | 47.0 | 60.8 |

* mAP is the average of APs at IOU threshoulds of 0.5-0.95 (10 segments). Evaluations are on either testing or validation split of the corresponding datasets.

#### For benchmarking (2D only)
| name | #params | training data | download | AP50 | AP75 | mAP |
| --- | --- | --- | --- | --- | --- | --- |
| small-2dL | 40M | LiveCell | [model](https://huggingface.co/jiyuuchc/lacss3-small-livecell/resolve/main/lacss3-small-l)| 84.3 | 61.1 | 57.4 |
| small-2dC | 40M | Cellpose+Cyto2 | [model](https://huggingface.co/jiyuuchc/lacss3-small-cellpose/resolve/main/lacss3-small-c) |87.6 | 62.0 | 56.4 |
| small-2dN | 40M | NIPS challenge |[model](https://huggingface.co/jiyuuchc/lacss3-small-nips/resolve/main/lacss3-small-n)| 84.6 | 64.8 | 57.3 |

#### Deployment

You can now deploy the pretrain models as GRPC server:

```sh
   python -m lacss.deploy.remote_server --modelpath=<model_file_path>
```

For a GUI client see the [Trackmate-Lacss](https://github.com/jiyuuchc/TrackMate-Lacss) project, which provides a FIJI/ImageJ plugin to perform cell segmentation/tracking in an interactive manner.


## Why LACSS?

  * multi-modality: works on both 2D (multichannel) images and 3D image stacks.

  * Speed: Inference time of the base model (150M parameters) is < 200 ms on GPU for an 1024x1024x3 image. We achieve this by desigining an end-to-end algorithm and aggressively eliminate CPU-dependent post-processings.

  * Point-supervised traing: Lacss is a multi-task model with a separate "localization" head (beside the segmentation head) predicting cell locations. This also means that you can train/fine-tune cell-segmentation models using only point labels. See [refernces](#references) for details.

## Give It A Try:

#### Gradio Demo: try your own images (2D only)
  * [Demo site](https://huggingface.co/spaces/yulabuchc/lacss-space)

#### Colabs
  * [Supervised Training ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/lacss1/notebooks/train_with_segmentation_label.ipynb)
  * [With point label + mask ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/lacss1/notebooks/train_with_point_and_mask.ipynb)
  * [With point label only ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/lacss1/notebooks/train_with_point_label.ipynb)

  * [Inference ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/inference.ipynb)

## Documentation
> [API documentation](https://jiyuuchc.github.io/lacss/)

## References
- [IEEE TMI (2023) doi:10.1109/TMI.2023.3312988](https://ieeexplore.ieee.org/document/10243149)
- [Communications Biology 6,232 (2023)](https://www.nature.com/articles/s42003-023-04608-5)

