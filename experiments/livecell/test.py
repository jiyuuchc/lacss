#!/usr/bin/env python3

import sys
import os
import json
import argparse
from os.path import join
import numpy as np
import tensorflow as tf
import lacss
import data
from tqdm import tqdm

from lacss.metrics import AJI
min_score = 0.4

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

def run_test(data_path, log_dir, checkpoint='model_weights'):
    print('evaluating...')
    ds_test = data.livecell_dataset_from_tfrecord(join(data_path, 'test.tfrecord'))

    with open(join(log_dir, 'config.json')) as f:
        model_cfg = json.load(f)
    model_cfg['test_max_output']=3000
    model = lacss.models.LacssModel.from_config(model_cfg)
    model.load_weights(join(log_dir, checkpoint)).expect_partial()
    print(f'model loaded from {checkpoint}')
    print(json.dumps(model.get_config(), sort_keys=True, indent=2))

    thresholds = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
    cell_aji = {}
    cell_APs = {}
    for c in range(8):
        cell_aji[c] = lacss.metrics.AJI()
        cell_APs[c] = lacss.metrics.MeanAP(thresholds)
    mAP = lacss.metrics.MeanAP(thresholds)
    for x in tqdm(ds_test):
        xx = lacss.data.parse_test_data_func(x)
        y = model(xx)

        scores = y['pred_location_scores'][0]
        ious = lacss.ops.mask_ious(x['mask_indices'], x['bboxes'], y['instance_output'][0], y['instance_coords'][0])
        ious = tf.transpose(ious)
        c = x['cell_type'].numpy()
        cell_APs[c].update_state(ious, scores)
        mAP.update_state(ious, scores)

        patches = y['instance_output'][0][scores > min_score]
        coords = y['instance_coords'][0][scores > min_score]
        cell_aji[c].update(x['mask_indices'], x['bboxes'], patches, coords)

    print('APs...')
    for c in range(8):
        result = cell_APs[c].result()
        result_str = format_array(result)
        print(f'{c}: {result_str}')
    all_result = mAP.result()
    result_str = format_array(all_result)
    print(f'all: {result_str}')

    print('FNRs...')
    for c in range(8):
        m = cell_APs[c]
        fnr_str = format_array(fnrs(m))
        print(f'{c}: {fnr_str}')
    fnr_str = format_array(fnrs(mAP))
    print(f'all: {fnr_str}')

    print('AJIs...')
    all_u = 0
    all_c = 0
    for c in range(8):
        all_u += cell_aji[c].u
        all_c += cell_aji[c].c
        result = cell_aji[c].result()
        print(f'{c}: {result}')
    print(f'all: {all_u/all_c}')

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Test livecell model')
    parser.add_argument('datapath', type=str, help='Data dir of tfrecord files')
    parser.add_argument('logpath', type=str, help='Log dir with model config and weights')
    parser.add_argument('--checkpoint', type=str, default='model_weight', help='Model checkpoint name')
    args = parser.parse_args()

    run_test(args.datapath, args.logpath, checkpoint=args.checkpoint)
