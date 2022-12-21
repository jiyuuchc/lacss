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
import tqdm

from lacss.metrics import AJI
min_score = 0.4

def format_array(arr):
    s = [f'{v:.4f}' for v in arr]
    s = ',\t'.join(s)
    return s

def fnrs(m):
    fnrs=[]
    for indicators in m._np_mean_aps.indicator_list:
        tp = np.count_nonzero(np.concatenate(indicators))
        recall = tp / m._np_mean_aps.cell_counts
        fnr = 1-recall
        fnrs.append(fnr)
    return fnrs

def run_test(data_path, log_dir, checkpoint='model_weights'):
    print('evaluating...')
    ds_test = data.livecell_dataset_from_tfrecord(join(data_path, 'test.tfrecord'))

    with open(join(log_dir, 'config.json')) as f:
        model_cfg = json.load(f)
    model_cfg['test_max_output']=3000
    # model_cfg['test_min_score']=min_score
    model = lacss.models.LacssModel.from_config(model_cfg)
    model.load_weights(join(log_dir, checkpoint)).expect_partial()
    print(f'model loaded from {checkpoint}')

    thresholds = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
    cell_APs = {}
    aji = {}
    mAP = lacss.metrics.MaskMeanAP(thresholds)
    for x in tqdm.tqdm(ds_test):
        xx = lacss.data.parse_test_data_func(x)
        y = model(xx)

        scores = y['pred_location_scores'][0]
        patches = y['instance_output'][0]
        coords = y['instance_coords'][0]
        pred_bboxes = lacss.ops.bboxes_of_patches(patches, coords)
        pred = (patches, coords, pred_bboxes)

        gt_bboxes=x['bboxes']
        gt_mi=x['mask_indices']
        gt = (gt_mi,gt_bboxes)

        c = x['cell_type'].numpy()
        if not c in cell_APs:
            cell_APs[c] = lacss.metrics.MaskMeanAP(thresholds)
            aji[c] = AJI()
        cell_APs[c].update_state(gt, pred, scores)
        mAP.update_state(gt, pred, scores)

        gt_mi = tf.gather(gt_mi, scores > min_score)
        gt_bboxes = tf.gather(gt_bboxes, scores > min_score)
        patches = tf.gather(patches, scores > min_score)
        coords = tf.gather(coords, scores > min_score)
        aji[c].update(gt_mi, gt_bboxes, patches, coords)

    cell_types = list(cell_APs.keys())
    cell_types.sort()

    print('APs...')
    for c in cell_types:
        result = cell_APs[c].result()
        result_str = format_array(result)
        print(f'{c}: {result_str}')
    all_result = mAP.result()
    result_str = format_array(all_result)
    print(f'all: {result_str}')

    print('FNRs...')
    for c in cell_types:
        m = cell_APs[c]
        fnr_str = format_array(fnrs(m))
        print(f'{c}: {fnr_str}')
    fnr_str = format_array(fnrs(mAP))
    print(f'all: {fnr_str}')

    print('AJIs...'')
    for c in cell_types:
        result = aji[c].result()
        print(f'{c}: {result}')

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Test livecell model')
    parser.add_argument('datapath', type=str, help='Data dir of tfrecord files')
    parser.add_argument('logpath', type=str, help='Log dir with model config and weights')
    parser.add_argument('--checkpoint', type=str, default='model_weight', help='Model checkpoint name')
    args = parser.parse_args()

    run_test(args.datapath, args.logpath, checkpoint=args.checkpoint)
