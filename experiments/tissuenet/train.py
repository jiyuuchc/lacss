#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import tensorflow as tf

import sys
from os.path import join
import lacss
from skimage.measure import regionprops
from tqdm  import tqdm

layers = tf.keras.layers

def tissue_net_gen_fn(data_path):
    X = np.load(join(data_path, 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'y.npy'), mmap_mode='r+')
    platforms = np.load(join(data_path, 'platform_list.npy'))
    tissues = np.load(join(data_path, 'tissue_list.npy'))
    for x, y, pf, t in zip(X, Y, platforms, tissues):
        img = x.astype('float32')
        label_in_ch0 = np.argmax(np.count_nonzero(y, axis=(0,1))) == 0
        y = y[..., 0] if label_in_ch0 else y[..., 1]
        # y = y[..., 0]
        binary_mask = (y > 0).astype('float32')
        locs = [prop['centroid'] for prop in regionprops(y)]
        mis = []
        mi_lengths= []
        bboxes = []
        for prop in regionprops(y):
            bboxes.append(prop['bbox'])
            mi = np.array(np.where(prop['image'])).transpose()
            mi = mi + bboxes[-1][:2]
            mi_lengths.append(mi.shape[0])
            mis.append(mi)
        
        bboxes = np.array(bboxes, dtype='float32')
        mis = tf.RaggedTensor.from_row_lengths(np.concatenate(mis), mi_lengths)

        yield {
            'image': img,
            'locations': locs,
            'binary_mask': binary_mask,
            'bboxes': bboxes,
            'mask_indices': mis,
            'platform': pf,
            'tissue': t,
        }

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
        x['img_id'] = 0
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
        print((ap50, ap75))

def run_training(args):
    data_path = args.datapath
    ds_train = tf.data.Dataset.from_generator(
        lambda: tissue_net_gen_fn(join(data_path, 'train')),
        output_signature = {
        'image': tf.TensorSpec([None, None, 2], dtype=tf.float32),
        'locations': tf.TensorSpec([None, 2], dtype=tf.float32),
        'binary_mask': tf.TensorSpec([None, None], dtype=tf.float32),
        'bboxes': tf.TensorSpec([None, 4], dtype=tf.float32),
        'mask_indices': tf.RaggedTensorSpec([None, None, 2], tf.int64, 1),
        'platform': tf.TensorSpec([], tf.string),
        'tissue': tf.TensorSpec([], tf.string),
        }
    )

    ds_val = tf.data.Dataset.from_generator(
        lambda: tissue_net_gen_fn(join(data_path, 'val')),
        output_signature = {
            'image': tf.TensorSpec([None, None, 2], dtype=tf.float32),
            'locations': tf.TensorSpec([None, 2], dtype=tf.float32),
            'binary_mask': tf.TensorSpec([None, None], dtype=tf.float32),
            'bboxes': tf.TensorSpec([None, 4], dtype=tf.float32),
            'mask_indices': tf.RaggedTensorSpec([None, None, 2], tf.int64, 1),
            'platform': tf.TensorSpec([], tf.string),
            'tissue': tf.TensorSpec([], tf.string),
        }
    )

    if args.config == "":
        model = lacss.models.LacssModel(
            detection_level=2,
            backbone='resnet_att',
            detection_head_conv_filters=[256,256,256,256],
            detection_head_fc_filters=[],
            instance_crop_size=128,
            train_max_output=800,
            test_max_output=400,
            test_min_score=0.4,
            test_pre_nms_topk=0,
            detection_nms_threshold=1.5,
            train_supervised=False,
            train_batch_size=1,
            )
    else:
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

    log_func = lambda epoch, _ : evaluation(model, ds_val.filter(lambda _: tf.random.uniform([])>=.9), args.logpath, epoch)
    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=args.logpath, write_graph=False),
            tf.keras.callbacks.ModelCheckpoint(filepath=join(args.logpath, 'chkpts-{epoch:02d}'), save_weights_only=True),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_func),
            ]

    cp = tf.train.latest_checkpoint(args.logpath)
    if cp is not None:
        model.load_weights(cp)
        init_epoch = int(cp.split('-')[-1])
    else:
        init_epoch = 0

    if init_epoch < 30:
        model.fit(ds_train, epochs=30, callbacks=callbacks, steps_per_epoch=3500//n_batch, initial_epoch=init_epoch)
        init_epoch = 30

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Train tissuenet model')
    parser.add_argument('datapath', type=str, help='Data dir of tfrecord files')
    parser.add_argument('logpath', type=str, help='Log dir for storing results')
    parser.add_argument('--config', type=str, default="", help='path to the model config file')   
    # parser.add_argument('--lpnlevel', type=int, default=3, help='LPN input feature level')
    args = parser.parse_args()

    try:
        os.makedirs(args.logpath)
    except:
        pass

    run_training(args)
