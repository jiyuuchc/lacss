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

def _parse_train_data_func(data, feature_level_scale, flip, scale_jitter, target_height, target_width):
    img = tf.cast(data['image'], tf.float32) / 255.0
    height, width, _ = img.shape

    # scale image so all cells are of similar pixel size
    scale = 1.0 / tf.gather(scale_values, data['cell_type'])
    if scale_jitter is not None:
        scale = scale * tf.random.uniform((), minval=scale_jitter[0], maxval=scale_jitter[1])
    scaled_size = tf.cast(tf.math.round([height*scale, width*scale]), tf.int32)
    img = tf.image.resize(img, scaled_size)

    # compute cell locations
    bboxes = tf.cast(data['bboxes'], tf.float32)
    y0 = (bboxes[:, 0] + bboxes[:, 2])/2*scale
    x0 = (bboxes[:, 1] + bboxes[:, 3])/2*scale

    # random_crop_or_pad to target sizes
    d_h = scaled_size[0]  - target_height
    d_w = scaled_size[1] - target_width
    if d_h >= 0:
        h0 = tf.random.uniform((), 0, d_h+1, tf.int32)
        img = tf.image.crop_to_bounding_box(img, h0, 0, target_height, scaled_size[1])
        y0 = y0 - tf.cast(h0, tf.float32)
    else:
        img = tf.image.pad_to_bounding_box(img, 0, 0, target_height, scaled_size[1])
    if d_w >=0:
        w0 = tf.random.uniform((), 0, d_w+1, tf.int32)
        img = tf.image.crop_to_bounding_box(img, 0, w0, target_height, target_width)
        x0 = x0 - tf.cast(w0, tf.float32)
    else:
        img = tf.image.pad_to_bounding_box(img, 0, 0, target_height, target_width)

    if flip:
        if tf.random.uniform(()) >= 0.5:
            img = tf.image.flip_left_right(img)
            x0 = tf.cast(target_width, tf.float32) - x0
        if tf.random.uniform(()) >= 0.5:
            img = tf.image.flip_up_down(img)
            y0 = tf.cast(target_height, tf.float32) - y0

    # remove out-of-bound locations
    locations = tf.stack([y0, x0], axis=-1)
    y_sel = tf.logical_and(y0>=0, y0<target_height)
    x_sel = tf.logical_and(x0>=0, x0<target_width)
    selections = tf.where(tf.logical_and(x_sel, y_sel))
    locations = tf.gather_nd(locations, selections)

    # generate regression targets
    # scaled_locations = locations / feature_level_scale
    # n_locs = tf.shape(scaled_locations)[0]
    # regression_scores = tf.scatter_nd(
    #     tf.cast(scaled_locations, tf.int32),
    #     tf.ones((n_locs,), tf.int32),
    #     shape=(target_height//feature_level_scale, target_width//feature_level_scale),
    # )
    #
    # if n_locs > 0:
    #     regression_scores = tf.expand_dims(regression_scores, -1)
    #     xc, yc = tf.meshgrid(tf.range(target_width//feature_level_scale), tf.range(target_height//feature_level_scale))
    #     mesh = tf.cast(tf.stack([yc, xc], axis=-1), tf.float32)
    #     distances = scaled_locations[:, None, None, :] - mesh
    #     distances_sq = distances * distances
    #     indices = tf.expand_dims(tf.argmin(distances_sq[..., 0] + distances_sq[..., 1], axis=0, output_type=tf.int32), 0)
    #     offsets_x = tf.experimental.numpy.take_along_axis(distances[..., 1], indices, 0)[0]
    #     offsets_y = tf.experimental.numpy.take_along_axis(distances[..., 0], indices, 0)[0]
    #     regression_offsets = tf.stack([offsets_y, offsets_x], axis=-1)
    # else:
    #     regression_offsets = -tf.ones([target_height//feature_level_scale, target_width//feature_level_scale, 2], tf.float32)

    return {
        'image': img,
        'locations': locations,
        # 'scores': regression_scores,
        # 'offsets': regression_offsets,
    }

def _parse_val_data_func(data, feature_level_scale):
    img = tf.cast(data['image'], tf.float32) / 255.0
    height, width, _ = img.shape

    # scale image so all cells are of similar pixel size
    scale = 1.0 / tf.gather(scale_values, data['cell_type'])
    scaled_size = tf.cast(tf.math.round([height*scale, width*scale]), tf.int32)
    img = tf.image.resize(img, scaled_size)

    # compute cell locations
    bboxes = tf.cast(data['bboxes'], tf.float32)
    y0 = (bboxes[:, 0] + bboxes[:, 2])/2*scale
    x0 = (bboxes[:, 1] + bboxes[:, 3])/2*scale
    locations = tf.stack([y0, x0], axis=-1)

    #pad to size of 32 x multiple
    new_height = (scaled_size[0] - 1) // 32 * 32 + 32
    new_width = (scaled_size[1] - 1) // 32 * 32 + 32
    img = tf.image.pad_to_bounding_box(img, 0, 0, new_height, new_width)

    return {
        'image': img,
        'locations': locations,
    }

def parse_train_data(dataset, feature_level_scale=8, flip=True, scale_jitter=None, target_height=512, target_width=704):
    return dataset.map(lambda x: _parse_train_data_func(x, feature_level_scale, flip, scale_jitter, target_height, target_width))

def parse_val_data(dataset, feature_level_scale=8):
    return dataset.map(lambda x: _parse_val_data_func(x, feature_level_scale))
