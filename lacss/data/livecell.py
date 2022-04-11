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
