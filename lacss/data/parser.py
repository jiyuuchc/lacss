import numpy as np
import tensorflow as tf


def crop_or_pad_to_target_size(image, target_height, target_width):
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    # n_ch = tf.shape(image)[2]

    d_h = height - target_height
    d_w = width - target_width
    if d_h > 0:
        h0 = tf.random.uniform((), 0, d_h + 1, tf.int32)
        image = tf.image.crop_to_bounding_box(image, h0, 0, target_height, width)
    else:
        h0 = 0
        image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, width)
    if d_w > 0:
        w0 = tf.random.uniform((), 0, d_w + 1, tf.int32)
        image = tf.image.crop_to_bounding_box(image, 0, w0, target_height, target_width)
    else:
        w0 = 0
        image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)

    image = tf.ensure_shape(image, [target_height, target_width, None])
    return image, h0, w0


def parse_train_data_func_full_annotation(
    data, augment=True, min_mask_area=16, target_height=512, target_width=512
):
    image = data["image"]
    masks = data["mask_indices"]
    mask_row_ids = masks.value_rowids()
    mask_values = masks.values

    target_width = int(target_width)
    target_height = int(target_height)

    image, h0, w0 = crop_or_pad_to_target_size(image, target_height, target_width)
    mask_values = mask_values - tf.cast([h0, w0], mask_values.dtype)
    if augment:
        if tf.random.uniform(()) >= 0.5:
            image = tf.image.flip_left_right(image)
            mask_values = mask_values * [1, -1] + [0, target_width - 1]
        if tf.random.uniform(()) >= 0.5:
            image = tf.image.flip_up_down(image)
            mask_values = mask_values * [-1, 1] + [target_height - 1, 0]

    valid_rows = tf.logical_and(
        tf.logical_and(mask_values[:, 0] >= 0, mask_values[:, 0] < target_height),
        tf.logical_and(mask_values[:, 1] >= 0, mask_values[:, 1] < target_width),
    )
    mask_values = tf.boolean_mask(mask_values, valid_rows)
    mask_row_ids = tf.boolean_mask(mask_row_ids, valid_rows)
    masks = tf.RaggedTensor.from_value_rowids(mask_values, mask_row_ids)
    mask_row_lengths = masks.row_lengths()
    valid_masks = mask_row_lengths >= min_mask_area
    masks = tf.ragged.boolean_mask(masks, valid_masks)

    locations = tf.cast(tf.reduce_mean(masks, axis=1), tf.float32)

    row_ids = masks.value_rowids() + 1
    labels = tf.zeros([target_height, target_width], row_ids.dtype)
    labels = tf.tensor_scatter_nd_update(labels, masks.values, row_ids)

    if augment and target_height == target_width:
        if tf.random.uniform([]) >= 0.5:
            image = tf.image.transpose(image)
            locations = locations[..., ::-1]
            labels = tf.transpose(labels)

    return {
        "image": image,
        "locations": locations,
        "mask_labels": labels,
    }


def parse_train_data_func(
    data, augment=True, size_jitter=None, target_height=512, target_width=512
):
    image = data["image"]
    binary_mask = tf.expand_dims(data["binary_mask"], -1)
    locations = data["locations"]

    img_and_label = tf.concat([image, tf.cast(binary_mask, tf.float32)], -1)
    if size_jitter is not None:
        h = tf.cast(tf.shape(img_and_label)[-3], tf.float32)
        w = tf.cast(tf.shape(img_and_label)[-2], tf.float32)
        scaling = tf.random.uniform([], size_jitter[0], size_jitter[1])
        img_and_label = tf.image.resize(
            img_and_label, [int(h * scaling), int(w * scaling)], antialias=True
        )
        locations = locations * scaling

    img_and_label, h0, w0 = crop_or_pad_to_target_size(
        img_and_label, target_height, target_width
    )
    locations = locations - tf.cast([h0, w0], tf.float32)

    if augment:
        if tf.random.uniform([]) >= 0.5:
            img_and_label = tf.image.flip_left_right(img_and_label)
            locations = locations * [1.0, -1.0] + [0, target_width]
        if tf.random.uniform([]) >= 0.5:
            img_and_label = tf.image.flip_up_down(img_and_label)
            locations = locations * [-1.0, 1.0] + [target_height, 0]

        if target_height == target_width:
            if tf.random.uniform([]) >= 0.5:
                img_and_label = tf.image.transpose(img_and_label)
                locations = locations[:, ::-1]

    image = img_and_label[..., :-1]
    if size_jitter is not None:
        binary_mask = tf.cast(img_and_label[..., -1] > 0.5, tf.float32)
    else:
        binary_mask = img_and_label[..., -1]

    # remove out-of-bound locations
    is_valid = tf.logical_and(
        tf.logical_and(locations[:, 0] > 1, locations[:, 0] < target_height - 1),
        tf.logical_and(locations[:, 1] > 1, locations[:, 1] < target_width - 1),
    )
    locations = tf.boolean_mask(locations, is_valid)

    # if augment:
    #     image = tf.image.random_brightness(image, 0.3)
    #     image = tf.image.random_contrast(image, 0.7, 1.3)

    return {
        "image": image,
        "locations": locations,
        "binary_mask": binary_mask,
    }


def parse_test_data_func(data, dim_multiple=32):
    image = data["image"]
    binary_mask = tf.expand_dims(data["binary_mask"], -1)
    bboxes = data["bboxes"]
    locations = data["locations"]
    masks = data["mask_indices"]

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    target_height = (height - 1) // dim_multiple * dim_multiple + dim_multiple
    target_width = (width - 1) // dim_multiple * dim_multiple + dim_multiple

    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)
    binary_mask = tf.cast(
        tf.image.pad_to_bounding_box(binary_mask, 0, 0, target_height, target_width),
        tf.float32,
    )
    binary_mask = tf.squeeze(binary_mask, -1)

    scaling = data["scaling"] if "scaling" in data else tf.constant(1.0, tf.float32)

    row_ids = masks.value_rowids() + 1
    labels = tf.zeros([target_height, target_width], row_ids.dtype)
    labels = tf.tensor_scatter_nd_update(labels, masks.values, row_ids)

    return {
        "image": image,
        "bboxes": bboxes,
        "locations": locations,
        "img_id": data["img_id"],
        "scaling": scaling,
        "binary_mask": binary_mask,
        "mask_indices": masks,
        "mask_labels": labels,
    }


def get_locations_from_labels(label):
    from skimage.measure import regionprops

    rps = regionprops(label)
    locs = [tf.constant(rp["centroid"]) for rp in rps]

    return tf.stack(locs)


def augment(data, size_jitter=None, target_height=512, target_width=512):

    if not isinstance(data, tuple):
        inputs = data
        labels = {}
    else:
        inputs, labels = data

    image = tf.cast(inputs["image"], "float32")

    if len(tf.shape(image)) == 2:
        image = image[:, :, None]

    assert len(tf.shape(image)) == 3

    if "gt_locations" in inputs:
        gt_locations = tf.cast(inputs["gt_locations"], "float32")
    elif "mask_labels" in labels:
        gt_locations = get_locations_from_labels(labels["mask_labels"])
    else:
        raise ValueError(
            "Training data requires gt_locations in the input or mask_labels in labels."
        )

    if "mask" in labels:
        mask = tf.cast(labels["mask"], "float32")

        if len(tf.shape(mask)) == 2:
            mask = mask[:, :, None]

        img_and_label = tf.concat([image, mask], -1)

    else:

        img_and_label = image

    if size_jitter is not None:
        h = tf.cast(tf.shape(img_and_label)[-3], tf.float32)
        w = tf.cast(tf.shape(img_and_label)[-2], tf.float32)
        scaling = tf.random.uniform([], size_jitter[0], size_jitter[1])
        img_and_label = tf.image.resize(
            img_and_label, [int(h * scaling), int(w * scaling)], antialias=True
        )
        gt_locations = gt_locations * scaling

    img_and_label, h0, w0 = crop_or_pad_to_target_size(
        img_and_label, target_height, target_width
    )
    gt_locations = gt_locations - tf.cast([h0, w0], tf.float32)

    if tf.random.uniform([]) >= 0.5:
        img_and_label = tf.image.flip_left_right(img_and_label)
        gt_locations = gt_locations * [1.0, -1.0] + [0, target_width]
    if tf.random.uniform([]) >= 0.5:
        img_and_label = tf.image.flip_up_down(img_and_label)
        gt_locations = gt_locations * [-1.0, 1.0] + [target_height, 0]

    if target_height == target_width and tf.random.uniform([]) >= 0.5:
        img_and_label = tf.image.transpose(img_and_label)
        gt_locations = gt_locations[:, ::-1]

    is_valid = tf.logical_and(
        tf.logical_and(locations[:, 0] > 1, locations[:, 0] < target_height - 1),
        tf.logical_and(locations[:, 1] > 1, locations[:, 1] < target_width - 1),
    )
    locations = tf.boolean_mask(locations, is_valid)

    if "mask" in labels:
        labels["mask"] = tf.cast(img_and_label[..., -1] > 0.5, tf.float32)
        image = img_and_label[..., :-1]
    else:
        image = img_and_label

    image = tf.image.random_contrast(image, 0.6, 1.4)
    image = tf.image.random_brightness(image, 0.3)

    inputs["image"] = image
    inputs["gt_locations"] = gt_locations

    return inputs, labels
