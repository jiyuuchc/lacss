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

layers = tf.keras.layers

def evaluation(model, ds, log_dir, epoch):
<<<<<<< HEAD
    print('evaluating...')
=======
>>>>>>> 2b41ac383498b4a5bb175da7938b088009f984c4
    test_log_dir = join(log_dir, 'validation')
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    mask_mAP = {}
    for x in ds:
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
            ap50 = mask_mAP[k].result()[0]
            tf.summary.scalar(f'mask_ap_{k}', ap50, step=epoch)
            print(f'[{k}]: AP50 = {ap50}')

def run_training(args):
    ds_train = data.livecell_dataset_from_tfrecord(join(args.datapath, 'train.tfrecord'))
    ds_val = data.livecell_dataset_from_tfrecord(join(args.datapath, 'val.tfrecord'))

    n_batch = 1
    parse_func = lambda x: lacss.data.parse_train_data_func(x, size_jitter=(0.85, 1.1), target_height=544, target_width=704)
    ds_train = ds_train.map(parse_func).filter(lambda s: tf.size(s['locations']) > 0).repeat()
    ds_train = ds_train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=n_batch))

    model = lacss.models.LacssModel(
        detection_level=args.lpnlevel,
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

    with open(join(args.logpath, 'config.json'), 'w') as f:
        json.dump(model.get_config(), f)

<<<<<<< HEAD
    log_func = lambda epoch, _ : evaluation(model, ds_val, args.logpath, epoch)
    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=args.logpath, write_graph=False),
            tf.keras.callbacks.ModelCheckpoint(filepath=join(args.logpath, 'chkpts-{epoch:02d}'), save_weights_only=True),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_func),
            ]
    model.fit(ds_train, epochs=30, callbacks=callbacks, steps_per_epoch=3500//n_batch, initial_epoch=0)

    ds_train = ds_train.filter(lambda x: x['cell_type']==args.celltype)
    model.fit(ds_train, epoch=45, callbacks=callbacks, steps_per_epoch=500/n_batch, initial_epoch=30)
=======
    log_func = lambda epoch, _ : evaluation(model, ds_val, log_dir, epoch)
    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False),
            tf.keras.callbacks.ModelCheckpoint(filepath=join(log_dir, 'chkpts-{epoch:02d}'), save_weights_only=True),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_func),
            ]
    model.fit(ds_train, epochs=30, callbacks=callbacks, steps_per_epoch=3500//n_batch, initial_epoch=0)
>>>>>>> 2b41ac383498b4a5bb175da7938b088009f984c4

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Train livecell model')
    parser.add_argument('datapath', type=str, help='Data dir of tfrecord files')
    parser.add_argument('logpath', type=str, help='Log dir for storing results')
    parser.add_argument('--celltype', type=int, default=0, help='Cell type 0-7')
    parser.add_argument('--lpnlevel', type=int, default=3, help='LPN input feature level')
    args = parser.parse_args()

    try:
        os.makedirs(args.logpath)
    except:
        pass

    run_training(args)

