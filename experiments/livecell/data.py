import random
import zipfile
from functools import partial
from os.path import join

import imageio
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

from lacss.train import TFDatasetAdapter

# import cv2


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
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


# split_table = {'train':0, 'val': 1, 'test':2}
cell_type_table = {
    "A172": 0,
    "BT474": 1,
    "BV2": 2,
    "Huh7": 3,
    "MCF7": 4,
    "SHSY5Y": 5,
    "SKOV3": 6,
    "SkBr3": 7,
}
cell_size_scales = {
    "A172": 1.0,
    "BT474": 0.65,
    "BV2": 0.50,
    "Huh7": 1.30,
    "MCF7": 0.50,
    "SHSY5Y": 1.0,
    "SKOV3": 1.30,
    "SkBr3": 0.75,
}
# scale_values = tf.constant(list(cell_size_scales.values()))


def livecell_data_gen(path, split):
    import cv2

    if split == "train":
        coco = COCO(
            annotation_file=join(
                path, "annotations", "LIVECell", "livecell_coco_train.json"
            )
        )
        img_path = join(path, "images", "livecell_train_val_images")
    elif split == "val":
        coco = COCO(
            annotation_file=join(
                path, "annotations", "LIVECell", "livecell_coco_val.json"
            )
        )
        img_path = join(path, "images", "livecell_train_val_images")
    elif split == "test":
        coco = COCO(
            annotation_file=join(
                path, "annotations", "LIVECell", "livecell_coco_test.json"
            )
        )
        img_path = join(path, "images", "livecell_test_images")
    else:
        raise ValueError("split needs to be one of the 'train', 'val' or 'test'.")

    ids = coco.getImgIds().copy()
    random.shuffle(ids)
    for imgid in ids:
        imginfo = coco.imgs[imgid]
        filename = imginfo["file_name"]
        cell_type = filename.split("_")[0]
        cell_type_id = cell_type_table[cell_type]

        img = imageio.imread(join(img_path, cell_type, filename)).astype("float32")
        img = img / 255.0

        height, width = img.shape
        scaling = 1.0 / cell_size_scales[cell_type]
        target_height = round(height * scaling)
        target_width = round(width * scaling)

        scaled_img = tf.image.resize(
            img[..., None], (target_height, target_width), antialias=scaling < 1.0
        )

        mask_indices = []
        bboxes = []
        row_lengths = []
        for ann_id in coco.getAnnIds(imgIds=imgid):
            box = np.array(coco.anns[ann_id]["bbox"]) * scaling
            seg = np.array(coco.anns[ann_id]["segmentation"]).reshape([1, -1, 2])
            mask = cv2.fillPoly(
                np.zeros([target_height, target_width], "uint8"),
                np.round(seg * scaling).astype("int32"),
                1,
            )
            mask = mask[
                int(box[1]) : int(box[1] + box[3] + 1),
                int(box[0]) : int(box[0] + box[2] + 1),
            ]
            indices = np.stack(np.where(mask)).transpose() + [int(box[1]), int(box[0])]
            mask_indices.append(indices)
            row_lengths.append(indices.shape[0])
            bboxes.append([box[1], box[0], box[1] + box[3], box[0] + box[2]])
        mis = np.concatenate(mask_indices)
        n_pixels = mis.shape[0]
        mis = tf.RaggedTensor.from_row_lengths(mis, row_lengths)

        locations = tf.cast(tf.reduce_mean(mis, axis=1), tf.float32)
        bboxes = np.array(bboxes).astype("float32")
        binary_mask = tf.scatter_nd(
            mis.values, tf.ones([n_pixels], tf.float32), (target_height, target_width)
        )
        binary_mask = tf.cast(binary_mask > 0.5, tf.float32)

        yield {
            "img_id": imgid,
            "image": scaled_img,
            "mask_indices": mis,
            "locations": locations,
            "binary_mask": binary_mask,
            "bboxes": bboxes,
            "scaling": scaling,
        }


def livecell_dataset(data_path, split):
    ds = tf.data.Dataset.from_generator(
        lambda: livecell_data_gen(data_path, split),
        output_signature={
            "img_id": tf.TensorSpec([], dtype=tf.int64),
            "image": tf.TensorSpec([None, None, 1], dtype=tf.float32),
            "mask_indices": tf.RaggedTensorSpec([None, None, 2], tf.int64, 1),
            "locations": tf.TensorSpec([None, 2], dtype=tf.float32),
            "binary_mask": tf.TensorSpec([None, None], dtype=tf.float32),
            "bboxes": tf.TensorSpec([None, 4], dtype=tf.float32),
            "scaling": tf.TensorSpec([], dtype=tf.float32),
        },
    )
    return ds


def parse_coco_record(coco, imgid, data_dir):
    imginfo = coco.imgs[imgid]
    filename = imginfo["file_name"]
    cell_type = filename.split("_")[0]
    cell_type_id = cell_type_table[cell_type]

    img = imageio.imread(join(data_dir, cell_type, filename)).astype("float32")
    height, width = img.shape
    img = img / 255.0

    bboxes = []
    mis = []
    rls = []
    locs = []
    scaling = 1.0 / cell_size_scales[cell_type]
    target_height = round(height * scaling)
    target_width = round(width * scaling)

    img = tf.image.resize(
        img[..., None], (target_height, target_width), antialias=scaling < 1.0
    ).numpy()

    ann_ids = coco.getAnnIds(imgIds=imgid)
    for ann_id in ann_ids:
        ann = coco.anns[ann_id]
        bbox = ann["bbox"]
        bbox = np.array([bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]])
        bboxes.append(bbox)
    n_boxes = len(bboxes)
    bboxes = np.array(bboxes, dtype="float32") * scaling
    selected = tf.image.non_max_suppression(
        bboxes, np.ones([n_boxes], "float32"), n_boxes, iou_threshold=0.75
    ).numpy()
    bboxes = bboxes[(selected,)]
    ann_ids = np.array(ann_ids)[(selected,)]

    for ann_id in ann_ids:
        ann = coco.anns[ann_id]
        mask = coco.annToMask(ann)
        if scaling != 1.0:
            mask = tf.image.resize(
                mask[..., None], (target_height, target_width), antialias=scaling < 1.0
            ).numpy()
            mask = mask.squeeze(axis=-1)
        mi = np.stack(np.where(mask >= 0.5), axis=-1)
        mis.append(mi)
        rls.append(mi.shape[0])
        locs.append(mi.mean(axis=0))
    # bboxes = np.array(bboxes, dtype='float32') * scaling
    locs = np.array(locs, dtype="float32")
    mis = np.concatenate(mis, dtype="int64")
    rls = np.array(rls, dtype="int64")
    # binary_mask = (skimage.transform.resize(binary_mask, (target_height, target_width)) >= 0.5).astype('uint8')
    binary_mask = np.zeros_like(img, dtype="uint8")
    binary_mask[tuple(mis.transpose())] = 1

    data = tf.train.Features(
        feature={
            "img_id": _int64_feature(imgid),
            "cell_type": _int64_feature(cell_type_id),
            "scaling": _float_feature(scaling),
            "image": _serialize_array(img),
            "locations": _serialize_array(locs),
            "bboxes": _serialize_array(bboxes),
            "binary_mask": _serialize_array(binary_mask),
            "mask_indices": _serialize_array(mis),
            "mask_indice_row_lengths": _serialize_array(rls),
        }
    )

    return tf.train.Example(features=data)


def create_tfrecord(data_path, extract_zip=False):
    """write downloaded data into tfrecord
    Args:
      data_path: the path_str where is downloaded data are. Output will be in the same directory
    """
    if extract_zip:
        print("extracting images from the zip...")
        with zipfile.ZipFile(join(data_path, "images.zip"), "r") as zf:
            zf.extractall(join(data_path, "images"))
        print("done")

    val_filename = join(data_path, "val.tfrecord")
    writer = tf.io.TFRecordWriter(val_filename)

    coco = COCO(
        annotation_file=join(
            data_path, "annotations", "LIVECell", "livecell_coco_val.json"
        )
    )
    ids = coco.getImgIds().copy()
    random.shuffle(ids)
    for k, id in enumerate(ids):
        seq = parse_coco_record(
            coco, id, join(data_path, "images", "livecell_train_val_images")
        )
        if seq:
            writer.write(seq.SerializeToString())
        if (k + 1) % 100 == 0:
            print(f"Wrote {k+1} records")
    print(f"Wrote {k} validation records")

    train_filename = join(data_path, "train.tfrecord")
    writer = tf.io.TFRecordWriter(train_filename)

    coco = COCO(
        annotation_file=join(
            data_path, "annotations", "LIVECell", "livecell_coco_train.json"
        )
    )
    ids = coco.getImgIds().copy()
    random.shuffle(ids)
    for k, id in enumerate(ids):
        seq = parse_coco_record(
            coco, id, join(data_path, "images", "livecell_train_val_images")
        )
        if seq:
            writer.write(seq.SerializeToString())
        if (k + 1) % 100 == 0:
            print(f"Wrote {k+1} records")
    print(f"Wrote {k} training records")

    writer.close()

    test_filename = join(data_path, "test.tfrecord")
    writer = tf.io.TFRecordWriter(test_filename)

    coco = COCO(
        annotation_file=join(
            data_path, "annotations", "LIVECell", "livecell_coco_test.json"
        )
    )
    ids = coco.getImgIds().copy()
    random.shuffle(ids)
    for k, id in enumerate(ids):
        seq = parse_coco_record(
            coco, id, join(data_path, "images", "livecell_test_images")
        )
        if seq:
            writer.write(seq.SerializeToString())
        if (k + 1) % 100 == 0:
            print(f"Wrote {k+1} records")
    print(f"Wrote {k} testing records")

    writer.close()


def tfr_parse_record(record):
    data = tf.io.parse_single_example(
        record,
        features={
            "img_id": tf.io.FixedLenFeature([], tf.int64),
            "cell_type": tf.io.FixedLenFeature([], tf.int64),
            "scaling": tf.io.FixedLenFeature([], tf.float32),
            "image": tf.io.FixedLenFeature([], tf.string),
            "bboxes": tf.io.FixedLenFeature([], tf.string),
            "locations": tf.io.FixedLenFeature([], tf.string),
            "binary_mask": tf.io.FixedLenFeature([], tf.string),
            "mask_indices": tf.io.FixedLenFeature([], tf.string),
            "mask_indice_row_lengths": tf.io.FixedLenFeature([], tf.string),
        },
    )

    img = tf.ensure_shape(
        tf.io.parse_tensor(data["image"], tf.float32), (None, None, 1)
    )
    # img = tf.expand_dims(img, -1)

    locations = tf.io.parse_tensor(data["locations"], tf.float32)
    # locations = locations[:, :2]
    locations = tf.ensure_shape(locations, [None, 2])
    binary_mask = tf.ensure_shape(
        tf.io.parse_tensor(data["binary_mask"], tf.uint8), (None, None, 1)
    )

    bboxes = tf.ensure_shape(tf.io.parse_tensor(data["bboxes"], tf.float32), (None, 4))

    mask_indice_values = tf.io.parse_tensor(data["mask_indices"], tf.int64)
    # mask_indice_values = mask_indice_values[:, :2]
    mask_indice_values = tf.ensure_shape(mask_indice_values, [None, 2])
    mask_indice_row_lengths = tf.ensure_shape(
        tf.io.parse_tensor(data["mask_indice_row_lengths"], tf.int64), (None,)
    )
    mask_indices = tf.RaggedTensor.from_row_lengths(
        mask_indice_values, mask_indice_row_lengths
    )

    return {
        "img_id": data["img_id"],
        "cell_type": data["cell_type"],
        "scaling": data["scaling"],
        "image": img,
        "locations": locations,
        "binary_mask": binary_mask[:, :, 0],
        "bboxes": bboxes,
        "mask_indices": mask_indices,
    }


def livecell_dataset_from_tfrecord(tfrpath):
    """
    Args:
        pathname: the tfrecord path
    Return:
        ds_train, ds_test:  a tuple of datasets if splits were supplied.
    """

    ds = tf.data.TFRecordDataset(tfrpath).map(tfr_parse_record)
    return ds


def train_parser_supervised(inputs):
    from lacss.data import parse_train_data_func_full_annotation

    inputs = parse_train_data_func_full_annotation(
        inputs, target_height=544, target_width=544
    )

    image = inputs["image"]
    gt_locations = inputs["locations"]
    mask_labels = tf.cast(inputs["mask_labels"], tf.float32)

    x_data = dict(
        image=image,
        gt_locations=gt_locations,
    )
    y_data = dict(
        gt_labels=mask_labels,
    )

    return x_data, y_data


def train_parser_semisupervised(inputs):
    from lacss.data import parse_train_data_func

    cell_type = tf.cast(inputs["cell_type"], tf.float32)
    inputs = parse_train_data_func(
        inputs, size_jitter=(0.85, 1.15), target_height=544, target_width=544
    )

    image = inputs["image"]
    gt_locations = inputs["locations"]
    mask = inputs["binary_mask"]

    image = tf.image.random_contrast(image, 0.6, 1.4)
    image = tf.image.random_brightness(image, 0.3)

    x_data = dict(
        image=image,
        gt_locations=gt_locations,
        category=tf.reshape(cell_type, [1]),
    )
    y_data = dict(
        gt_mask=mask,
    )

    return x_data, y_data


def train_data(
    datapath, n_buckets, batchsize, *, supervised=False, cell_type=-1, coco=False
):
    if not coco:
        ds_train = livecell_dataset_from_tfrecord(join(datapath, "train.tfrecord"))

    else:
        from lacss.data import dataset_from_coco_annotations

        ds_train = dataset_from_coco_annotations(
            join(datapath, "annotations", "LIVECell", "livecell_coco_train.json"),
            join(datapath, "images", "livecell_train_val_images"),
            [520, 704, 1],
        ).cache(join(datapath, "cache_train"))

    ds_train = ds_train.repeat()
    if cell_type >= 0:
        ds_train = ds_train.filter(lambda x: x["cell_type"] == cell_type)

    if supervised:
        ds_train = ds_train.map(train_parser_supervised)
    else:
        ds_train = ds_train.map(train_parser_semisupervised)

    ds_train = ds_train.filter(lambda x, _: tf.size(x["gt_locations"]) > 0)

    ds_train = ds_train.bucket_by_sequence_length(
        element_length_func=lambda x, _: tf.shape(x["gt_locations"])[0],
        bucket_boundaries=list(np.arange(1, n_buckets + 1) * (4096 // n_buckets) + 1),
        bucket_batch_sizes=(max(batchsize, 1),) * (n_buckets + 1),
        padding_values=-1.0,
        pad_to_bucket_boundary=True,
    )

    if batchsize <= 0:
        ds_train = ds_train.unbatch()

    return TFDatasetAdapter(ds_train, steps=-1).get_dataset()


def pad_to(x):
    s = tf.shape(x)[0]
    sp = (s - 1) // 1024 * 1024 + 1024
    x = tf.pad(x, [[0, sp - s], [0, 0]], constant_values=-1.0)
    return x


def val_parser(inputs, supervised):
    cell_type = tf.cast(inputs["cell_type"], tf.float32)
    img = inputs["image"]
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    h1 = (h - 1) // 32 * 32 + 32 - h
    w1 = (w - 1) // 32 * 32 + 32 - w
    img = tf.pad(img, [[0, h1], [0, w1], [0, 0]])

    masks = inputs["mask_indices"]
    row_ids = masks.value_rowids() + 1
    labels = tf.zeros([h + h1, w + w1], row_ids.dtype)
    labels = tf.tensor_scatter_nd_update(labels, masks.values, row_ids)

    if supervised:
        x_data = dict(
            image=img,
        )
    else:
        x_data = dict(
            image=img,
            category=tf.reshape(cell_type, [1]),
        )

    y_data = dict(
        gt_boxes=pad_to(inputs["bboxes"]),
        gt_locations=pad_to(inputs["locations"]),
        gt_labels=labels,
    )

    return x_data, y_data


def val_data(
    datapath, *, supervised=False, cell_type=-1, coco=False, split="val", batch=True
):
    if not coco:
        tfr_name = split + ".tfrecord"
        ds_val = livecell_dataset_from_tfrecord(join(datapath, tfr_name))

    else:
        from lacss.data import dataset_from_coco_annotations

        json_file = f"livecell_coco_{split}.json"
        img_path = (
            "livecell_train_val_images" if split == "val" else "livecell_test_images"
        )
        ds_val = dataset_from_coco_annotations(
            join(datapath, "annotations", "LIVECell", json_file),
            join(datapath, "images", img_path),
            [520, 704, 1],
        ).cache(join(datapath, f"cache_{split}"))

    if cell_type >= 0:
        ds_val = ds_val.filter(lambda x: x["cell_type"] == cell_type)

    ds_val = ds_val.map(partial(val_parser, supervised=supervised))

    if batch:
        ds_val = ds_val.batch(1)

    return TFDatasetAdapter(ds_val).get_dataset()


def test_parser(inputs, supervised):
    cell_type = tf.cast(inputs["cell_type"], tf.float32)

    img = inputs["image"]
    gt_mask = inputs["binary_mask"]
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    h1 = (h - 1) // 32 * 32 + 32 - h
    w1 = (w - 1) // 32 * 32 + 32 - w
    img = tf.pad(img, [[0, h1], [0, w1], [0, 0]])
    gt_mask = tf.pad(gt_mask, [[0, h1], [0, w1]])

    mis = inputs["mask_indices"]
    mi_ds = mis.value_rowids()[:, None]
    mi_vs = tf.cast(mis.values, mi_ds.dtype)
    mask_indices = tf.concat([mi_ds, mi_vs], axis=-1)

    if supervised:
        x_data = dict(
            image=img,
        )
    else:
        x_data = dict(
            image=img,
            category=tf.reshape(cell_type, [1]),
        )

    y_data = dict(
        gt_boxes=inputs["bboxes"],
        gt_locations=inputs["locations"],
        gt_mask=gt_mask,
        gt_mask_indices=mask_indices,
    )

    return x_data, y_data


def test_data(datapath, *, supervised=False, cell_type=-1, coco=False):
    if not coco:
        ds_val = livecell_dataset_from_tfrecord(join(datapath, "test.tfrecord"))

    else:
        from lacss.data import dataset_from_coco_annotations

        ds_test = dataset_from_coco_annotations(
            join(datapath, "annotations", "LIVECell", "livecell_coco_test.json"),
            join(datapath, "images", "livecell_test_images"),
            [520, 704, 1],
        ).cache(join(datapath, f"cache_test"))

    if cell_type >= 0:
        ds_val = ds_val.filter(lambda x: x["cell_type"] == cell_type)

    ds_val = ds_val.map(partial(test_parser, supervised=supervised))

    return TFDatasetAdapter(ds_val).get_dataset()
