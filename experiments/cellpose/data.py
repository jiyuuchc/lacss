import numpy as np
import tensorflow as tf
from os.path import join
import cv2
from skimage.measure import regionprops

_target_size = 32

def cellpose_data_gen(path, n_records):
    for k in range(n_records):
        imgfile = join(path, f'{k:03d}_img.png')
        maskfile = join(path, f'{k:03d}_masks.png')
        with open(imgfile,'rb') as f:
            img = tf.io.decode_image(f.read())
        img = tf.cast(img, tf.float32) / 255.0
        with open(maskfile,'rb') as f:
            mask = tf.io.decode_png(f.read(), channels=1, dtype=tf.uint16).numpy()
        n_masks = mask.max()

        props = regionprops(mask[:,:,0])
        bboxes = np.array([p['bbox'] for p in props])
        scaling = _target_size / np.median(np.concatenate((bboxes[:,3] - bboxes[:,1], bboxes[:,2] - bboxes[:, 0])))

        h = int(mask.shape[0] * scaling + .5)
        w = int(mask.shape[1] * scaling + .5)
        img = tf.image.resize(img, (h,w))

        mis = []
        row_lengths = []
        for k, p in enumerate(props):
            crop = p['image'].astype('uint8')
            c, a = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = np.round((np.array(c[0]) + bboxes[k, (1,0)]) * scaling)
            mask_new = cv2.fillPoly(np.zeros((h,w), dtype='uint8'), [c.astype('int')], 1)
            mi = np.stack(np.where(mask_new)).transpose()
            mis.append(mi)
            row_lengths.append(mi.shape[0])
        mis = np.concatenate(mis)
        n_pixels = mis.shape[0]
        mis = tf.RaggedTensor.from_row_lengths(mis, row_lengths)

        locations = tf.cast(tf.reduce_mean(mis, axis=1), tf.float32)
        bboxes = (bboxes * scaling).astype('float32')
        binary_mask = tf.scatter_nd(mis.values, tf.ones([n_pixels], tf.float32), (h,w))
        binary_mask = tf.cast(binary_mask > 0.5, tf.float32)

        yield {
            'img_id': k,
            'image':img,
            'mask_indices': mis,
            'locations': locations,
            'binary_mask': binary_mask,
            'bboxes': bboxes
        }

def cellpose_dataset(data_path, n_records):
    ds = tf.data.Dataset.from_generator(
        lambda : cellpose_data_gen(data_path, n_records),
        output_signature={
            'img_id': tf.TensorSpec([], dtype=tf.int64),
            'image': tf.TensorSpec([None,None,3], dtype=tf.float32),
            'mask_indices': tf.RaggedTensorSpec([None, None, 2], tf.int64, 1),
            'locations': tf.TensorSpec([None,2], dtype=tf.float32),
            'binary_mask': tf.TensorSpec([None,None], dtype=tf.float32),
            'bboxes': tf.TensorSpec([None,4], dtype=tf.float32),
        }
    )
    return ds
