# LACSS
LACSS is a deep-learning model for 2D/3D single-cell segmentation from microscopy images.
```sh
   pip install lacss[cuda12]
```

## Models checkpoints
#### Multi-modality (2D + 3D)
| name | #params | download | mAP LiveCell* | mAP Cellpose* | mAP NIPS* | ovule (3D)* | platynereis (3D)* |
| :--- | --- | --- | :---: | :---: | :---: | :---: | :---: | 
| small | 60M | [model](https://huggingface.co/jiyuuchc/lacss3-small/resolve/main/lacss3-small)| 56.3 | 52.0 | 54.2 | 44.4 | 56.7 |
| base | 152M | [model](https://huggingface.co/jiyuuchc/lacss3-base/resolve/main/lacss3-base)| 57.1 | 56.0 | 62.9 | 47.0 | 60.8 |
| base-e | 304M | [model](https://huggingface.co/jiyuuchc/lacss3-base-e/resolve/main/lacss3-base-e) | 57.4 | 58.3 | 65.7 | 49.8 | 61.9 |

* mAP is the average of APs at IOU threshoulds of 0.5-0.95 (10 segments). Evaluations are on either testing or validation split of the corresponding datasets.

#### For benchmarking (2D only)
| name | #params | training data | download | AP50 | AP75 | mAP |
| --- | --- | --- | --- | --- | --- | --- |
| small-2dL | 40M | LiveCell | [model](https://huggingface.co/jiyuuchc/lacss3-small-livecell/resolve/main/lacss3-small-l)| 84.3 | 61.1 | 57.4 |
| small-2dC | 40M | Cellpose+Cyto2 | [model](https://huggingface.co/jiyuuchc/lacss3-small-cellpose/resolve/main/lacss3-small-c) |87.6 | 62.0 | 56.4 |
| small-2dN | 40M | NIPS challenge |[model](https://huggingface.co/jiyuuchc/lacss3-small-nips/resolve/main/lacss3-small-n)| 84.6 | 64.8 | 57.3 |

#### Deployment

You can deploy the models as an [GRPC](https://grpc.io/) server using the [biopb.image](https://github.com/jiyuuchc/biopb) protocol:

```sh
   python -m lacss.deploy.remote_server --modelpath=<model_file_path>
```

#### Public server

The Lacss public GRPC server is available here: `lacss.biopb.org:443`

The server is running the base model supporting both 2d and 3d segmentation.

#### For end user 

 - [Trackmate-Lacss](https://github.com/jiyuuchc/TrackMate-Lacss) is the recommended GUI client for FIJI users. This plugin integrate with TrackMate for interactive cell segmentation and cell tracking.
 - [napari-biopb](https://github.com/jiyuuchc/napari-biopb) is recommended for napari users. 
 - For setting up your analysis pipeline programmatically, see this example [notebook](https://github.com/jiyuuchc/lacss/blob/main-jax/notebooks/grpc.ipynb).

## Why LACSS?

  * Multi-modality: works on both 2D (multichannel) images and 3D image stacks.

  * Speed: optimized for GPU due to the end-to-end design and the elimination of CPU-dependent post-processings.

  * Point-supervised training: Lacss is a multi-task model with a separate "localization" head (besides the segmentation head) predicting cell locations. This also means that you can train/fine-tune cell-segmentation using only point labels. See [references](#references) for details.

## Give It A Try:

#### Gradio Demo: try your own images (2D only)
  * [Demo site](https://biopb.org/demo/)

#### Colabs
##### Inference
  * [Inference ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/inference.ipynb)
  * [GRPC call ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/grpc.ipynb)

##### Train
  * [Supervised Training ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_segmentation_label.ipynb)
  * [With point label only ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_point_label.ipynb)


## Documentation
> [API documentation](https://jiyuuchc.github.io/lacss/)

## References
- [IEEE TMI (2023) doi:10.1109/TMI.2023.3312988](https://ieeexplore.ieee.org/document/10243149)
- [Communications Biology 6,232 (2023)](https://www.nature.com/articles/s42003-023-04608-5)

