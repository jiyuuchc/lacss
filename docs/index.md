LACSS is a deep-learning model for single-cell segmentation from microscopy images.

### Why LACSS?
LACSS is designed to utilize point labels for model training. You have three options:

| Method | Data(left) / Label(right)|
| --- | --- |
| Point | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_1.png" width="300"> |
| Point + Mask | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_2.png" width="300"> |
| Segmentation | <img src="https://github.com/jiyuuchc/lacss/raw/main-jax/.github/images/label_scheme_3.png" width="300"> |

You can also combined these labels in any way you want.

