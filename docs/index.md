LACSS is a deep-learning model for single-cell segmentation from microscopy images.

### Why LACSS?
LACSS is designed to utilize point labels for model training. You have three options:

| Method | Data(left) / Label(right)|
| --- | --- |
| Point | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_1.png" width="300"> |
| Point + Mask | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_2.png" width="300"> |
| Segmentation | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_3.png" width="300"> |

You can of course also combined these labels in any way you want.

### Give it a try:
* Model training
  - [Supervised Training ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_segmentation_label.ipynb)
  - [With point label + mask ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_point_and_mask.ipynb)
  - [With point label only ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/train_with_point_label.ipynb)

* [Inference ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main-jax/notebooks/inference.ipynb)

