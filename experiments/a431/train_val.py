#!/usr/bin/env python3
import sys
import os
import json
import argparse
from os.path import join
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import lacss

layers = tf.keras.layers

def train_parser(x):
    x = lacss.data.parse_train_data_func(x, size_jitter=(0.9, 1.1), target_height=512, target_width=512)
    if tf.random.uniform([]) >=0.5:
         x['image'] = tf.image.transpose(x['image'])
         x['binary_mask'] = tf.image.transpose(x['binary_mask'])
         x['locations'] = x['locations'][:,::-1]
    return x


def evaluation(model, ds, log_dir, epoch):
    print('evaluating...')
    test_log_dir = join(log_dir, 'validation')
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    mask_AP = lacss.metrics.MaskMeanAP([.5, .75])
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

        mask_AP.update_state(gt, pred, scores)

    with test_summary_writer.as_default():
        ap50, ap75 = mask_AP.result()
        tf.summary.scalar(f'mask_ap50', ap50, step=epoch)
        tf.summary.scalar(f'mask_ap75', ap75, step=epoch)

def run_training(args):
    data_path = args.datapath
    log_dir = args.logpath

    ds_train = lacss.data.dataset_from_simple_annotations(join(data_path, 'train.json'), join(data_path, 'train'), [None,None,1])

    imgfiles = [join(data_path, 'test', f'img_{k+1:04d}.tif') for k in range(25)]
    maskfiles = [join(data_path, 'test', f'masks_{k+1:04d}.tif') for k in range(25)]

    ds_val =lacss.data.dataset_from_img_mask_pairs(imgfiles, maskfiles, [None, None, 1])

    print(ds_train.element_spec)

    with open(args.config) as f:
        model_cfg = json.load(f)
    model = lacss.models.LacssModel.from_config(model_cfg)

    n_batch = model.get_config()['train_batch_size']
    ds_train = ds_train.map(train_parser).repeat()
    ds_train = ds_train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=n_batch))

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)

    with open(join(args.logpath, 'config.json'), 'w') as f:
        json.dump(model.get_config(), f)

    log_func = lambda epoch, _ : evaluation(model, ds_val, args.logpath, epoch)
    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=args.logpath, write_graph=False),
            tf.keras.callbacks.ModelCheckpoint(filepath=join(args.logpath, 'chkpts-{epoch:02d}'), save_weights_only=True),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_func),
            ]

    cp = args.transfer
    if cp != "":
        model.load_weights(cp)
 
    model.fit(ds_train, epochs=20, callbacks=callbacks, steps_per_epoch=500//n_batch)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Train A431 model')
    parser.add_argument('datapath', type=str, help='Data dir of tfrecord files')
    parser.add_argument('logpath', type=str, help='Log dir for storing results')
    parser.add_argument('--config', type=str, help='path to the model config file')   
    parser.add_argument('--transfer', type=str, default="", help='Transfer learning checkpoint')
    args = parser.parse_args()

    try:
        os.makedirs(args.logpath)
    except:
        pass

    run_training(args)
