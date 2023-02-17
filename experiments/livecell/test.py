#!/usr/bin/env python3

import json
from os.path import join
from tqdm import tqdm

import numpy as np
import elegy as eg
import lacss
import jax
jnp =jax.numpy

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

try:
    from . import data
except:
    import data

import typer
app = typer.Typer(pretty_exceptions_enable=False)

def format_array(arr):
    s = [f'{v:.4f}' for v in arr]
    s = ',\t'.join(s)
    return s

@app.command()
def run_test(
    datapath: str,
    checkpoint: str,
):

    print('evaluating...')
    ds_test = data.livecell_dataset_from_tfrecord(datapath, 'test.tfrecord')

    model = eg.model.model_base.load(checkpoint)
    print(f'model loaded from {checkpoint}')

    thresholds = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
    mask_APs = {}
    for c in range(8):
        mask_APs[c] = lacss.metrics.MeanAP(thresholds)
    mask_mAP = lacss.metrics.MeanAP(thresholds)

    thresholds = [0.1, 0.2, 0.5, 1.0]
    loi_APs = {}
    for c in range(8):
        loi_APs[c] = lacss.metrics.MeanAP(thresholds)
    loi_mAP = lacss.metrics.MeanAP(thresholds)

    model = model.eval()
    for inputs in tqdm(ds_test):
        c = inputs['cell_type'].numpy()
        inputs = lacss.data.parse_test_data_func(inputs)
        image = jnp.repeat(inputs['image'].numpy(), 1, axis=-1)
        label = jnp.array(inputs['mask_labels'])

        preds = model.predict_on_batch(image[None, ...])
        preds = jax.tree_map(lambda x: x[0], preds)
        scores = np.array(preds['pred_scores'])
        valid = scores >= 0
        valid_scores = scores[valid]

        ious = lacss.ops.iou_patches_and_labels(preds, label)

        mask_APs[c].update_state(jnp.array(ious)[valid], valid_scores)
        mask_mAP.update_state(jnp.array(ious)[valid], valid_scores)

        gt_locs = inputs['locations'].numpy()
        pred_locs = np.array(preds['pred_locations'])[valid]
        dist2 = ((pred_locs[:,None,:] - gt_locs) ** 2).sum(axis=-1)
        sm = 1.0 / np.sqrt(dist2)
        loi_APs[c].update_state(sm, valid_scores)
        loi_mAP.update_state(sm, valid_scores)

    print('LOI APs...')
    for c in range(8):
        result = loi_APs[c].result()
        result_str = format_array(result)
        print(f'{c}: {result_str}')
    all_result = loi_mAP.result()
    result_str = format_array(all_result)
    print(f'all: {result_str}')

    print('Mask APs...')
    for c in range(8):
        result = mask_APs[c].result()
        result_str = format_array(result)
        print(f'{c}: {result_str}')
    all_result = mask_mAP.result()
    result_str = format_array(all_result)
    print(f'all: {result_str}')

if __name__ =="__main__":
    app()
