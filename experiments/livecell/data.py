from pycocotools.coco import COCO
from os.path import join
import random
import zipfile
import imageio
import skimage.transform

import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return _bytes_feature(array)

split_table = {'train':0, 'val': 1, 'test':2}
cell_type_table = {'A172': 0, 'BT474': 1, 'BV2': 2, 'Huh7': 3, 'MCF7': 4, 'SHSY5Y': 5, 'SKOV3': 6, 'SkBr3': 7}

cell_size_scales = {
    'A172': 1.0,
    'BT474': 0.65,
    'BV2': 0.50,
    'Huh7': 1.0,
    'MCF7': 0.50,
    'SHSY5Y': 0.65,
    'SKOV3': 1.30,
    'SkBr3': 0.75,
    }
scale_values = tf.constant(list(cell_size_scales.values()))

def parse_coco_record(coco, imgid, data_dir):
    imginfo = coco.imgs[imgid]
    filename = imginfo['file_name']
    cell_type = filename.split('_')[0]
    cell_type_id = cell_type_table[cell_type]

    img = imageio.imread(join(data_dir, cell_type, filename)).astype('float32')
    height, width = img.shape
    img = img / 255.0

    bboxes = []
    mis = []
    rls = []
    locs = []
    for ann_id in coco.getAnnIds(imgIds=imgid):
        ann = coco.anns[ann_id]
        bbox = ann['bbox']
        bbox = np.array([bbox[1], bbox[0], bbox[1]+bbox[3], bbox[0]+bbox[2]])
        bboxes.append(bbox)
        mask = coco.annToMask(ann)
        mi = np.stack(np.where(mask>0), axis=-1)
        mis.append(mi)
        rls.append(mi.shape[0])
        locs.append(mi.mean(axis=0))

    scaling = 1.0 / cell_size_scales[cell_type]
    target_height = round(height * scaling)
    target_width = round(width * scaling)

    bboxes = np.array(bboxes, dtype='float32') * scaling
    locs = np.array(locs, dtype='float32') * scaling

    mis = np.concatenate(mis, dtype='int64')
    rls = np.array(rls, dtype='int64')

    binary_mask = np.zeros_like(img)
    binary_mask[tuple(mis.transpose())] = 1

    img = skimage.transform.resize(img, (target_height, target_width))
    binary_mask = (skimage.transform.resize(binary_mask, (target_height, target_width)) >= 0.5).astype('uint8')

    data = tf.train.Features(feature = {
        'img_id': _int64_feature(imgid),
        'cell_type': _int64_feature(cell_type_id),
        'scaling': _float_feature(scaling),
        'image': _serialize_array(img),
        'locations': _serialize_array(locs),
        'bboxes': _serialize_array(bboxes),
        'binary_mask': _serialize_array(binary_mask),
        'mask_indices': _serialize_array(mis),
        'mask_indice_row_lengths': _serialize_array(rls),
    })

    return tf.train.Example(features=data)

def create_tfrecord(data_path, extract_zip=False):
    ''' write downloaded data into tfrecord
    Args:
      data_path: the path_str where is downloaded data are. Output will be in the same directory
    '''
    if extract_zip:
        print('extracting images from the zip...')
        with zipfile.ZipFile(join(data_path, 'images.zip'), 'r') as zf:
            zf.extractall(join(data_path, 'images'))
        print('done')

    val_filename = join(data_path, 'val.tfrecord')
    writer = tf.io.TFRecordWriter(val_filename)

    coco = COCO(annotation_file=join(data_path, 'annotations', 'LIVECell', 'livecell_coco_val.json'))
    ids = coco.getImgIds().copy()
    random.shuffle(ids)
    for k, id in enumerate(ids):
        seq = parse_coco_record(coco, id, join(data_path, 'images', 'livecell_train_val_images'))
        if seq:
            writer.write(seq.SerializeToString())
        if (k+1)%100 == 0:
            print(f'Wrote {k+1} records')
    print(f'Wrote {k} validation records')

    train_filename = join(data_path, 'train.tfrecord')
    writer = tf.io.TFRecordWriter(train_filename)

    coco = COCO(annotation_file=join(data_path, 'annotations', 'LIVECell', 'livecell_coco_train.json'))
    ids = coco.getImgIds().copy()
    random.shuffle(ids)
    for k, id in enumerate(ids):
        seq = parse_coco_record(coco, id, join(data_path, 'images', 'livecell_train_val_images'))
        if seq:
            writer.write(seq.SerializeToString())
        if (k+1)%100 == 0:
            print(f'Wrote {k+1} records')
    print(f'Wrote {k} training records')

    writer.close()

    test_filename = join(data_path, 'test.tfrecord')
    writer = tf.io.TFRecordWriter(test_filename)

    coco = COCO(annotation_file=join(data_path, 'annotations', 'LIVECell', 'livecell_coco_test.json'))
    ids = coco.getImgIds().copy()
    random.shuffle(ids)
    for k, id in enumerate(ids):
        seq = parse_coco_record(coco, id, join(data_path, 'images', 'livecell_test_images'))
        if seq:
            writer.write(seq.SerializeToString())
        if (k+1)%100 == 0:
            print(f'Wrote {k+1} records')
    print(f'Wrote {k} testing records')

    writer.close()

def tfr_parse_record(record):
    data = tf.io.parse_single_example(
        record,
        features = {
            'img_id': tf.io.FixedLenFeature([], tf.int64),
            'cell_type': tf.io.FixedLenFeature([], tf.int64),
            'scaling': tf.io.FixedLenFeature([], tf.float32),
            'image': tf.io.FixedLenFeature([], tf.string),
            'bboxes': tf.io.FixedLenFeature([], tf.string),
            'locations': tf.io.FixedLenFeature([], tf.string),
            'binary_mask': tf.io.FixedLenFeature([], tf.string),
            'mask_indices': tf.io.FixedLenFeature([], tf.string),
            'mask_indice_row_lengths': tf.io.FixedLenFeature([], tf.string),
        },
    )

    img = tf.ensure_shape(tf.io.parse_tensor(data['image'], tf.float32), (None, None))
    img = tf.expand_dims(img, -1)

    locations = tf.ensure_shape(tf.io.parse_tensor(data['locations'], tf.float32), (None, 2))
    binary_mask = tf.ensure_shape(tf.io.parse_tensor(data['binary_mask'], tf.uint8), (None, None))

    bboxes = tf.ensure_shape(tf.io.parse_tensor(data['bboxes'], tf.float32), (None, 4))
    mask_indice_values = tf.ensure_shape(tf.io.parse_tensor(data['mask_indices'], tf.int64), (None, 2))
    mask_indice_row_lengths = tf.ensure_shape(tf.io.parse_tensor(data['mask_indice_row_lengths'], tf.int64), (None,))
    mask_indices = tf.RaggedTensor.from_row_lengths(mask_indice_values, mask_indice_row_lengths)

    return {
        'img_id': data['img_id'],
        'cell_type': data['cell_type'],
        'scaling': data['scaling'],
        'image': img,
        'locations': locations,
        'binary_mask': binary_mask,
        'bboxes': bboxes,
        'mask_indices': mask_indices,
        }

def livecell_dataset_from_tfrecord(tfrpath):
    '''
    Args:
        pathname: the tfrecord path
    Return:
        ds_train, ds_test:  a tuple of datasets if splits were supplied.
    '''

    ds = tf.data.TFRecordDataset(tfrpath).map(tfr_parse_record)
    return ds
