## LACSS

_LACSS is a deep-learning model for single-cell segmentation from microscopy images._ 

References: 

- [IEEE TMI doi:10.1109/TMI.2023.3312988](https://arxiv.org/abs/2304.10671)
- [Communications Biology 6,232 (2023)](https://www.nature.com/articles/s42003-023-04608-5)

### Installation
```
pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install lacss
```

### Why LACSS?
LACSS is designed to utilize point labels for model training. You have three options:

| Method | Data(left) / Label(right)|
| --- | --- |
| Point | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_1.png" width="300"> |
| Point + Mask | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_2.png" width="300"> |
| Segmentation | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_3.png" width="300"> |

You can of course also combined these labels in any way you want.

### What is included?

- A library for training LACSS model and performing inference
- A few pretrained models as transfer learning starting point
- SMC-based cell tracking utility for people interested in cell tracking

### How to generate point label?

If your data include nuclei counter-stain, the easist way to generate point label for your image is to use a [blob detection](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html) algorithm on the nuclei images:

![](https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/blob_detection.png)

### Give it a try:
* Model training
  * [Supervised Training ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_segmentation_label.ipynb)
  * [With point label + mask ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_point_and_mask.ipynb)
  * [With point label only ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_point_label.ipynb)

* [Inference ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/inference.ipynb)

### Documentation
  [API documentation](https://jiyuuchc.github.io/lacss/api/deploy/)
