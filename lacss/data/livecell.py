import zipfile
import numpy as np
import tensorflow as tf
import pycocotools.coco as COCO
import imageio

from os.path import join

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

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

split_table = {'train':0, 'val': 1, 'test':2}
cell_type_table = {'A172': 0, 'BT474': 1, 'BV2': 2, 'Huh7': 3, 'MCF7': 4, 'SHSY5Y': 5, 'SKOV3': 6, 'SkBr3': 7}

scales = {
    'A172': 1.000000,
    'BT474': 0.658023,
    'BV2': 0.351476,
    'Huh7': 1.047497,
    'MCF7': 0.500642,
    'SHSY5Y': 0.646213,
    'SKOV3': 1.282927,
    'SkBr3': 0.521694,
    }
scale_values = tf.constant(list(scales.values()))

def parse_record(coco, imgid, data_dir, split='train'):
    imginfo = coco.imgs[imgid]
    filename = imginfo['file_name']
    cell_type = filename.split('_')[0]
    img = imageio.imread(os.path.join(data_dir, cell_type, filename))

    context = tf.train.Features(feature = {
        'image': _bytes_feature(serialize_array(img)),
        'cell_type': _int64_feature(cell_type_table[cell_type]),
        'split': _int64_feature(split_table[split]),
    })

    bboxes = []
    masks = []
    for ann_id in coco.getAnnIds(imgIds=imgid):
        ann = coco.anns[ann_id]
        mask = coco.annToMask(ann)
        bbox = ann['bbox']
        bbox = np.array([bbox[1], bbox[0], bbox[1]+bbox[3], bbox[0]+bbox[2]])
        bbox = (bbox + 0.5).astype(np.int64)
        mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        bboxes.append(tf.train.Feature(int64_list=tf.train.Int64List(value=bbox)))
        masks.append(_bytes_feature(serialize_array(mask)))

    feature_lists = tf.train.FeatureLists(
        feature_list = {
            'bboxes': tf.train.FeatureList(feature=bboxes),
            'masks': tf.train.FeatureList(feature=masks),
        }
    )

    seq = tf.train.SequenceExample(context = context, feature_lists=feature_lists)
    return seq

def export_to_tfrecord(data_path):
    ''' write downloaded data into tfrecord
    Args:
      data_path: the path_str where is downloaded data are. Output will be in the same directory
    '''
    print('extracting images from the zip...')
    with zipfile.ZipFile(join(daa_path, image.zip), 'r') as zf:
        zp.extractall(join(data_path, 'images'))
    print('done')

    filename = join(data_dir, 'livecell.tfrecord')
    writer = tf.io.TFRecordWriter(filename)

    coco = COCO(annotation_file=join(data_path, 'annotations', 'LIVECell', 'livecell_coco_val.json'))
    for k, id in enumerate(coco.getImgIds()):
        seq = parse_record(coco, id, join(data_dir, 'livecell_train_val_images'), split='val')
        writer.write(seq.SerializeToString())

    print(f'Wrote {k} validation records')

    coco = COCO(annotation_file=join(data_path, 'annotations', 'LIVECell', 'livecell_coco_test.json'))
    for k, id in enumerate(coco.getImgIds()):
        seq = parse_record(coco, id, join(data_dir, 'image', 'livecell_test_images'), split='test')
        writer.write(seq.SerializeToString())
    print(f'Wrote {k} testing records')

    coco = COCO(annotation_file=join(data_path, 'annotations', 'LIVECell', 'livecell_coco_train.json'))
    for k, id in enumerate(coco.getImgIds()):
        seq = parse_record(coco, id, join(data_dir, 'image', 'livecell_train_val_images'), split='train')
        writer.write(seq.SerializeToString())
    print(f'Wrote {k} training records')

    writer.close()

def tfrecord_parse_fun(record):
    data = tf.io.parse_sequence_example(
        record,
        context_features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'cell_type': tf.io.FixedLenFeature([], tf.int64),
            'split': tf.io.FixedLenFeature([], tf.int64),
        },
        sequence_features = {
            'bboxes': tf.io.FixedLenSequenceFeature([4,], tf.int64),
            'masks': tf.io.FixedLenSequenceFeature([], tf.string)
        },
    )

    img = tf.ensure_shape(tf.io.parse_tensor(data[0]['image'], tf.uint8), (520,704))
    img = tf.reshape(img, (520, 704, 1))

    return {
        'image': img,
        'cell_type': data[0]['cell_type'],
        'split': data[0]['split'],
        'bboxes': data[1]['bboxes'],
        'masks': data[1]['masks'],
    }

def livecell_dataset_from_tfrecord(filename, splits=None):
    '''
    Args:
        filename: the tfrecord file pathname
        splits: None (default) or a list of strings of the specific splits requests
    Return:
        either a dataset object for all data, or a tuple of datasets if splits were supplied.
    '''

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(tfrecord_parse_fun)

    if splits is None:
        return dataset
    else:
      ds_list = [dataset.filter(lambda d: d['split'] == split_table[split]) for split in splits]
      return tuple(ds_list)

def _make_mask_label(masks, offsets, target_shape=(520, 704)):
    offsets = tf.cast(offsets, tf.int64)
    def mask2indices(inputs):
        mask, offset = inputs
        mask = tf.io.parse_tensor(mask, tf.uint8)
        indices = tf.where(mask>0)
        indices = indices + offset
        return indices
    indices = tf.map_fn(
        mask2indices,
        (masks, offsets),
        fn_output_signature=(tf.RaggedTensorSpec((None,2), tf.int64, 0)),
    )

    # max_size = tf.cast(target_shape,  tf.float32) - 1.0
    # indices = tf.cast(indices, tf.float32) * scaling_factor / max_size
    # indices = tf.cast(tf.round(tf.clip_by_value(indices, 0., 1.) * max_size), tf.int32)

    mask_label = tf.scatter_nd(indices.merge_dims(0,1), indices.value_rowids()+1, target_shape)
    return mask_label

def _parse_train_data_func(data, feature_level_scale, flip, scale_jitter, target_height, target_width):
    img = tf.cast(data['image'], tf.float32) / 255.0
    height, width, _ = img.shape

    # scale image so all cells are of similar pixel size
    scale = 1.0 / tf.gather(scale_values, data['cell_type'])
    if scale_jitter is not None:
        scale = scale * tf.random.uniform((), minval=scale_jitter[0], maxval=scale_jitter[1])
    scaled_height = int(height * scale + .5)
    scaled_width = int(width * scale + .5)
    img = tf.image.resize(img, (scaled_height, scaled_width))

    # compute cell location labels
    bboxes = tf.cast(data['bboxes'], tf.float32)
    y0 = (bboxes[:, 0] + bboxes[:, 2])/2*scale
    x0 = (bboxes[:, 1] + bboxes[:, 3])/2*scale
    mask_label = _make_mask_label(data['masks'], bboxes[:,0:2])
    binary_mask = tf.cast(mask_label>0, tf.int32)
    binray_mask = tf.image.resize(binary_mask[...,None], (scaled_height,scaled_width))
    img_and_label = tf.concat([img, binray_mask], -1)

    if target_height is not None and target_width is not None:
        # random_crop_or_pad to target sizes
        d_h = scaled_height  - target_height
        d_w = scaled_width - target_width
        if d_h >= 0:
            h0 = tf.random.uniform((), 0, d_h+1, tf.int32)
            img_and_label = tf.image.crop_to_bounding_box(img_and_label, h0, 0, target_height, scaled_width)
            y0 = y0 - tf.cast(h0, tf.float32)
        else:
            h0 = 0
            img_and_label = tf.image.pad_to_bounding_box(img_and_label, 0, 0, target_height, scaled_width)
        if d_w >=0:
            w0 = tf.random.uniform((), 0, d_w+1, tf.int32)
            img_and_label = tf.image.crop_to_bounding_box(img_and_label, 0, w0, target_height, target_width)
            x0 = x0 - tf.cast(w0, tf.float32)
        else:
            w0 = 0
            img_and_label = tf.image.pad_to_bounding_box(img_and_label, 0, 0, target_height, target_width)
    else:
        #pad to size of 64 x multiple
        target_height = (scaled_height - 1) // 64 * 64 + 64
        target_width = (scaled_width - 1) // 64 * 64 + 64
        img_and_label = tf.image.pad_to_bounding_box(img_and_label, 0, 0, target_height, target_width)

    if flip:
        if tf.random.uniform(()) >= 0.5:
            img_and_label = tf.image.flip_left_right(img_and_label)
            x0 = tf.cast(target_width, tf.float32) - x0
        if tf.random.uniform(()) >= 0.5:
            img_and_label = tf.image.flip_up_down(img_and_label)
            y0 = tf.cast(target_height, tf.float32) - y0

    # remove out-of-bound locations
    locations = tf.stack([y0, x0], axis=-1)
    y_sel = tf.logical_and(y0>=0, y0<tf.cast(target_height, tf.float32))
    x_sel = tf.logical_and(x0>=0, x0<tf.cast(target_width, tf.float32))
    selections = tf.where(tf.logical_and(x_sel, y_sel))
    locations = tf.gather_nd(locations, selections)

    return {
        'image': img_and_label[..., 0:1],
        'locations': locations,
        'scaling_factor': scale,
        'binary_label': tf.cast(img_and_label[..., 1] > 0.5, tf.int32),
    }

def _parse_val_data_func(data, feature_level_scale):
    img = tf.cast(data['image'], tf.float32) / 255.0
    height, width, _ = img.shape

    # scale image so all cells are of similar pixel size
    scale = 1.0 / tf.gather(scale_values, data['cell_type'])
    scaled_height = int(height * scale + .5)
    scaled_width = int(width * scale + .5)
    img = tf.image.resize(img, (scaled_height, scaled_width))

    # compute cell locations
    bboxes = tf.cast(data['bboxes'], tf.float32)
    y0 = (bboxes[:, 0] + bboxes[:, 2])/2*scale
    x0 = (bboxes[:, 1] + bboxes[:, 3])/2*scale
    locations = tf.stack([y0, x0], axis=-1)

    #pad to size of 64 x multiple
    new_height = (scaled_height - 1) // 64 * 64 + 64
    new_width = (scaled_width - 1) // 64 * 64 + 64
    img = tf.image.pad_to_bounding_box(img, 0, 0, new_height, new_width)

    # label
    mask_label = _make_mask_label(data['masks'], bboxes[:,0:2])

    return {
        'image': img,
        'locations': locations,
        'scaling_factor': scale,
        'mask_label': tf.cast(mask_label, tf.int32),
    }

def parse_train_data(dataset, feature_level_scale=8, flip=True, scale_jitter=None, target_height=None, target_width=None):
    return dataset.map(lambda x: _parse_train_data_func(x, feature_level_scale, flip, scale_jitter, target_height, target_width))

def parse_val_data(dataset, feature_level_scale=8):
    return dataset.map(lambda x: _parse_val_data_func(x, feature_level_scale))
