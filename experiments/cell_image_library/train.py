#!/usr/bin/env python

import sys
import os
import json
import argparse
from os.path import join

import numpy as np
import tensorflow as tf
import lacss
import tqdm

def evaluation(model, ds, log_dir, epoch):
    test_log_dir = join(log_dir, 'validation')
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    maskAP = lacss.metrics.MaskMeanAP([.5])
    for x in tqdm(ds):
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

        maskAP.update_state(gt, pred, scores)

    with test_summary_writer.as_default():
        tf.summary.scalar(f'mask_ap', maskAP.result()[0], step=epoch)

def train_parser(x):
    x = lacss.data.parse_train_data_func(x, size_jitter=(0.9, 1.1))
    if tf.random.uniform([]) >=0.5:
         x['image'] = tf.image.transpose(x['image'])
         x['binary_mask'] = tf.image.transpose(x['binary_mask'])
         x['locations'] = x['locations'][:,::-1]
    return x

def run_training(data_path, log_dir):
    imgfiles = [join(data_path, 'train', f'{k:03d}_img.png') for k in range(89)]
    maskfiles = [join(data_path, 'train', f'{k:03d}_masks.png') for k in range(89)]
    ds_train =lacss.data.dataset_from_img_mask_pairs(imgfiles, maskfiles)

    imgfiles = [join(data_path, 'test', f'{k:03d}_img.png') for k in range(11)]
    maskfiles = [join(data_path, 'test', f'{k:03d}_masks.png') for k in range(11)]
    ds_val =lacss.data.dataset_from_img_mask_pairs(imgfiles, maskfiles)

    n_batch = 1
    parse_func = train_parser
    ds_train = ds_train.map(parse_func).filter(lambda s: tf.size(s['locations']) > 0).repeat()
    ds_train = ds_train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=n_batch))

    model = lacss.models.LacssModel.from_config({})
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)

    with open(join(log_dir, 'config.json'), 'w') as f:
        json.dump(model.get_config(), f)

    log_func = lambda epoch, _ : evaluation(model, ds_val, log_dir, epoch)
    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False),
            tf.keras.callbacks.ModelCheckpoint(filepath=join(log_dir, 'chkpts-{epoch:02d}'), save_weights_only=True),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_func),
            ]
    model.fit(ds_train, epochs=20, callbacks=callbacks, steps_per_epoch=1000//n_batch, initial_epoch=0)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Train livecell model')
    parser.add_argument('data_path', type=str, help='Data dir of tfrecord files')
    parser.add_argument('log_path', type=str, help='Log dir for storing results')
    args = parser.parse_args()

    try:
        os.makedirs(args.log_path)
    except:
        pass

    run_training(args.data_path, args.log_path)
