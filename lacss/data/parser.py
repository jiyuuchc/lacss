import numpy as np
import tensorflow as tf

def scale_or_pad_to_target_size(image, target_height, target_width):
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    # n_ch = tf.shape(image)[2]

    d_h = height  - target_height
    d_w = width - target_width
    if d_h > 0:
        h0 = tf.random.uniform((), 0, d_h+1, tf.int32)
        image = tf.image.crop_to_bounding_box(image, h0, 0, target_height, width)
    else:
        h0 = 0
        image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, width)
    if d_w >0:
        w0 = tf.random.uniform((), 0, d_w+1, tf.int32)
        image = tf.image.crop_to_bounding_box(image, 0, w0, target_height, target_width)
    else:
        w0 = 0
        image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

    image = tf.ensure_shape(image, [target_height, target_width, None])
    return image, h0, w0

def parse_train_data_func_full_annotation(data, augment=True, min_mask_area=16, target_height=512, target_width=512):
    image = data['image']
    masks = data['mask_indices']
    mask_row_ids = masks.value_rowids()
    mask_values = masks.values

    target_width = int(target_width)
    target_height = int(target_height)

    image, h0, w0 = scale_or_pad_to_target_size(image, target_height, target_width)
    mask_values = mask_values - tf.cast([h0, w0], mask_values.dtype)
    if augment:
        if tf.random.uniform(()) >= 0.5:
            image = tf.image.flip_left_right(image)
            mask_values = mask_values * [1, -1] + [0, target_width]
        if tf.random.uniform(()) >= 0.5:
            image = tf.image.flip_up_down(image)
            mask_values = mask_values * [-1, 1] + [target_height, 0]

    valid_rows = tf.logical_and(
      tf.logical_and(mask_values[:,0]>=0, mask_values[:,0]<target_height),
      tf.logical_and(mask_values[:,1]>=0, mask_values[:,1]<target_width),
    )
    mask_values = tf.boolean_mask(mask_values, valid_rows)
    mask_row_ids = tf.boolean_mask(mask_row_ids, valid_rows)
    masks = tf.RaggedTensor.from_value_rowids(mask_values, mask_row_ids)
    mask_row_lengths = masks.row_lengths()
    valid_masks = mask_row_lengths >= min_mask_area
    masks = tf.ragged.boolean_mask(masks, valid_masks)

    locations = tf.cast(tf.reduce_mean(masks, axis=1), tf.float32)

    return {
        'image': image,
        'locations': locations,
        'mask_indices': masks,
    }

def parse_train_data_func(data, augment=True, size_jitter=None, target_height=512, target_width=512):
    image = data['image']
    binary_mask = tf.expand_dims(data['binary_mask'], -1)
    locations = data['locations']

    img_and_label = tf.concat([image, tf.cast(binary_mask, tf.float32)], -1)
    if size_jitter is not None:
        h = tf.cast(tf.shape(img_and_label)[-3], tf.float32)
        w = tf.cast(tf.shape(img_and_label)[-2], tf.float32)
        scaling = tf.random.uniform([], size_jitter[0], size_jitter[1])
        img_and_label = tf.image.resize(img_and_label, [int(h*scaling), int(w*scaling)], antialias=True)
        locations = locations * scaling

    img_and_label, h0, w0 = scale_or_pad_to_target_size(img_and_label, target_height, target_width)
    locations = locations - tf.cast([h0, w0], tf.float32)

    if augment:
        if tf.random.uniform(()) >= 0.5:
            img_and_label = tf.image.flip_left_right(img_and_label)
            locations = locations * [1.0, -1.0] + [0, target_width]
        if tf.random.uniform(()) >= 0.5:
            img_and_label = tf.image.flip_up_down(img_and_label)
            locations = locations * [-1.0, 1.0] + [target_height, 0]

    image = img_and_label[..., 0:1]
    if size_jitter is not None:
        binary_mask = tf.cast(img_and_label[..., 1:] > 0.5, tf.float32)
    else:
        binary_mask = img_and_label[..., 1:]

    #remove out-of-bound locations
    mask = tf.logical_and(
      tf.logical_and(locations[:,0]>1, locations[:,0]<target_height-1),
      tf.logical_and(locations[:,1]>1, locations[:,1]<target_width-1),
    )
    locations = tf.boolean_mask(locations, mask)

    # if augment:
    #     image = tf.image.random_brightness(image, 0.3)
    #     image = tf.image.random_contrast(image, 0.7, 1.3)

    return {
        'image': image,
        'locations': locations,
        'binary_mask': binary_mask,
    }

def parse_test_data_func(data, dim_multiple=64):
    image = data['image']
    binary_mask = tf.expand_dims(data['binary_mask'], -1)
    bboxes = data['bboxes']
    locations = data['locations']

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    target_height = (height - 1) // dim_multiple * dim_multiple + dim_multiple
    target_width =  (width - 1) // dim_multiple * dim_multiple + dim_multiple

    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)
    binary_mask = tf.cast(tf.image.pad_to_bounding_box(binary_mask, 0, 0, target_height, target_width), tf.float32)

    return {
        'image': image,
        'bboxes': bboxes,
        'locations': locations,
        'img_id': data['img_id'],
        'scaling': data['scaling'],
        'binary_mask': binary_mask,
        'mask_indices': data['mask_indices'],
    }
