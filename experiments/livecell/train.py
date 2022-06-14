#!/usr/bin/env python3

import sys
import os
import json
import argparse
from os.path import join
import numpy as np
import tensorflow as tf
import lacss
import tqdm
from .data import livecell_dataset_from_tfrecord

layers = tf.keras.layers

def evaluation(model, ds):
    test_log_dir = join(log_dir, 'validation')
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    mask_mAP = {}
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

        c = x['cell_type'].numpy()
        if not c in mask_mAP:
            mask_mAP[c] = lacss.metrics.MaskMeanAP([.5])
        mask_mAP[c].update_state(gt, pred, scores)

    with test_summary_writer.as_default():
        for k in mask_mAP:
            tf.summary.scalar(f'mask_ap_{k}', mask_mAP[k].result()[0], step=epoch)

def run_training(data_path, log_dir):
    ds_train = livecell_dataset_from_tfrecord(join(data_path, 'train.tfrecord'))
    ds_val = livecell_dataset_from_tfrecord(join(data_path, 'val.tfrecord'))

    n_batch = 1
    parse_func = lambda x: lacss.data.parse_train_data_func(x, size_jitter=(0.85, 1.1), target_height=544, target_width=704)
    ds_train = ds_train.map(parse_func).filter(lambda s: tf.size(s['locations']) > 0).repeat()
    ds_train = ds_train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=n_batch))

    model = lacss.models.LacssModel(
        backbone='resnet_att',
        train_max_output=600,
        test_max_output=2500,
        test_min_score=0.25,
        test_pre_nms_topk=0,
        train_supervised=False,
        train_batch_size=n_batch,
        )
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)

    with open(join(log_dir, 'config.json'), 'w') as f:
        json.dump(model.get_config(), f)

    log_func = lambda epoch, logs : evaluation(model, ds_val)
    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False),
            # tf.keras.callbacks.ModelCheckpoint(filepath=join(log_dir, 'chkpts-{epoch:02d}'), save_weights_only=True),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_func),
            ]
    model.fit(ds_train, epochs=20, callbacks=callbacks, steps_per_epoch=3500//n_batch, initial_epoch=0)

    model.save_weights(join(log_dir, 'model_weights'))

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
