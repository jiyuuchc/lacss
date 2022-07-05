# LACSS (Location assisted cell segmentation system)

_LACSS is a deep-learning model for single-cell segmentation built on weakly-supervised end-to-end learning._

Ref: [[https://www.biorxiv.org/content/10.1101/2022.07.03.498609v1]]

It is designed to utilize two types of weak annotations: (a) image-level segmentation, and (b) location-of-interests (LOIs). These annotatins are chosen because they can often be produced progammably using simple unsupervised algorithms from experimental data. Our goal is to build a streamlined annotation-training pipeline that requires no manual input from humans. Here's an example:

</br>


![pipeline](images/lacss1.png)

</br>

LACSS models are usually accurate enough to compete with fully-supervised models. Here are some benchmarks on the LIVECell dataset.

![benchmarks](images/lacss2.png)

</br>

You can try the demo notebook in colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyuuchc/lacss/blob/main/notebooks/demo.ipynb)







