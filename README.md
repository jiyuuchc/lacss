# LACSS

```sh
   pip install lacss
```
LACSS is a deep-learning model for single-cell segmentation from microscopy images. See references below:

- [IEEE TMI (2023) doi:10.1109/TMI.2023.3312988](https://ieeexplore.ieee.org/document/10243149)
- [Communications Biology 6,232 (2023)](https://www.nature.com/articles/s42003-023-04608-5)

## What's new (0.11)

You can now deploy the LACSS predictor as an GRPC server:

```sh
   python -m lacss.deploy.remote_server --modelpath=<model_file_path>
```

For a GUI client see the [Trackmate-Lacss](https://github.com/jiyuuchc/TrackMate-Lacss) project, which provides a FIJI/ImageJ plugin to perform cell segmentation/tracking in an interactive manner.


## Why LACSS?
> LACSS is designed to utilize **point labels** for model training. You have three options: (1) Label each cell with a single point, (2) label each cell with a single point and then label each image with a binary mask that covers all cells, or (3) Label each cell with a separate segmentation mask (as in standard supervised training). You can of course also combined these labels in any way you want.

| Method | Data(left) / Label(right)|
| --- | --- |
| Point | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_1.png" width="300"> |
| Point + Mask | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_2.png" width="300"> |
| Segmentation | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_3.png" width="300"> |


### How to generate point label?

> If your data include nuclei counter-stain, the easist way to generate point label for your image is to use a [blob detection](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html) algorithm on the nuclei images:

![](https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/blob_detection.png)

## Give It A Try:
### Model Training
  * [Supervised Training ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_segmentation_label.ipynb)
  * [With point label + mask ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_point_and_mask.ipynb)
  * [With point label only ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_point_label.ipynb)

### Model Inference
  * [Inference ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/inference.ipynb)

## Documentation
> [API documentation](https://jiyuuchc.github.io/lacss/)
