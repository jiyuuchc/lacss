#!/usr/bin/env python3

import json
import argparse
from os.path import join
import numpy as np
import elegy as eg
import tensorflow as tf
import lacss
from tqdm import tqdm
import jax
jnp =jax.numpy

tf.config.set_visible_devices([], 'GPU')

try:
    from . import data
except:
    import data

def format_array(arr):
    s = [f'{v:.4f}' for v in arr]
    s = ',\t'.join(s)
    return s

def fnrs(m):
    fnrs=[]
    for indicators in m.indicator_list:
        tp = np.count_nonzero(np.concatenate(indicators))
        recall = tp / m.cell_counts
        fnr = 1-recall
        fnrs.append(fnr)
    return fnrs

def run_test():
    log_dir = args.logpath
    data_path = args.datapath

    print('evaluating...')
    ds_test = data.livecell_dataset_from_tfrecord(join(data_path, 'test.tfrecord'))

    model = eg.model.model_base.load(join(log_dir, args.checkpoint))
    # model.module.detector._config_dict['test_max_output']=3000
    print(f'model loaded from {args.checkpoint}')

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

    print('FNRs...')
    for c in range(8):
        m = mask_APs[c]
        fnr_str = format_array(fnrs(m))
        print(f'{c}: {fnr_str}')
    fnr_str = format_array(fnrs(mask_mAP))
    print(f'all: {fnr_str}')

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Test livecell model')
    parser.add_argument('datapath', type=str, help='Data dir of tfrecord files')
    parser.add_argument('logpath', type=str, help='Log dir with model config and weights')
    parser.add_argument('--checkpoint', type=str, default='model_weight', help='Model checkpoint name')
    
    args = parser.parse_args()

    run_test()
