# LACSS

LACSS is a model for single-cell segmentation and cell-lineage tracking

Ref: https://www.nature.com/articles/s42003-023-04608-5

As a segmentation model, it can work similar to other instance segmentation models such as MaskRCNN. However, it also support end-to-end training with very weak supervisions: e.g (a) image-level segmentation, and (b) location-of-interests (LOIs). These annotatins are chosen because they can often be produced progammably using simple unsupervised algorithms from experimental data. Our goal is to build a streamlined annotation-training pipeline that requires no manual input from humans.

The segmentation model is used for down-stream cell-tracking task. The tracking logic is based on SMC (sequential Monte Carlo).

This particular version of LACSS is build on [Jax](https://github.com/google/jax) framework. Both the segmentation model and the tracking logic heavily utilize the composable transformation facility provided by JAX.



