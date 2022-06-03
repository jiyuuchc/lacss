import tensorflow as tf
import numpy as np
import json
import imageio
from pycocotools.coco import COCO
from os.path import join

def _coco_generator(annotation_file, image_path):
    coco = COCO(annotation_file=annotation_file)
    for imgid in coco.getImgIds():
        imginfo = coco.imgs[imgid]
        bboxes = []
        mis = []
        locs = []
        rls = []
        for ann_id in coco.getAnnIds(imgIds=imgid):
            ann = coco.anns[ann_id]
            bbox = ann['bbox']
            bbox = np.array([bbox[1], bbox[0], bbox[1]+bbox[3], bbox[0]+bbox[2]])
            bboxes.append(bbox)

            mask = coco.annToMask(ann)
            mi = np.stack(np.where(mask>=0.5), axis=-1)
            mis.append(mi)
            rls.append(mi.shape[0])

            locs.append(mi.mean(axis=0))

        bboxes = np.array(bboxes, dtype='float32')
        locs = np.array(locs, dtype='float32')
        mis = np.concatenate(mis, dtype='int64')
        rls = np.array(rls, dtype='int64')

        image = imageio.imread(join(image_path, imginfo['filename']))
        if len(image.shape) == 2:
            image = image[:,:,None]
        image = (image / 255.0).astype('float32')

        binary_mask = np.zeros(image.shape[:2], 'float32')
        binary_mask[tuple(mis.transpose())] = 1.0

        mis = tf.RaggedTensor.from_row_lengths(mis, rls)

        yield {
            'img_id': imgid,
            'image': image,
            'mask_indices': mis,
            'locations': locs,
            'binary_mask': binary_mask,
            'bboxes': bboxes,
        }

def dataset_from_coco_annotations(annotation_file, image_path, image_shape=[None, None, 3]):
    return tf.data.Dataset.from_generator(
        lambda : _coco_generator(annotation_file, image_path),
        output_signature={
            'img_id': tf.TensorSpec([], dtype=tf.int64),
            'image': tf.TensorSpec(image_shape, dtype=tf.float32),
            'mask_indices': tf.RaggedTensorSpec([None, None, 2], tf.int64, 1),
            'locations': tf.TensorSpec([None,2], dtype=tf.float32),
            'binary_mask': tf.TensorSpec([None,None], dtype=tf.float32),
            'bboxes': tf.TensorSpec([None,4], dtype=tf.float32),
        }
    )

def _simple_generator(annotation_file, image_path):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    for ann in annotations:
        image = imageio.imread(join(image_path, ann['image_file']))
        if len(image.shape) == 2:
            image = image[:,:,None]
        image = (image / 255.0).astype('float32')

        binary_mask = imageio.imread(join(image_path, ann['mask_file']))
        if len(binary_mask.shape) == 3:
            binary_mask = binary_mask[:,:,0]
        binary_mask = (binary_mask > 0).astype('float32')

        locations = np.array(ann['locations']).astype('float32')

        yield {
            'img_id': ann['id'],
            'image': image,
            'binary_mask': binary_mask,
            'locations': locations,
        }

def dataset_from_simple_annotations(annotation_file, image_path, image_shape=[None, None, 3]):
    return tf.data.Dataset.from_generator(
        lambda : _simple_generator(annotation_file, image_path),
        output_signature = {
            'img_id': tf.TensorSpec([], dtype=tf.int64),
            'image': tf.TensorSpec(image_shape, dtype=tf.float32),
            'locations': tf.TensorSpec([None,2], dtype=tf.float32),
            'binary_mask': tf.TensorSpec([None,None], dtype=tf.float32),
        }
    )
