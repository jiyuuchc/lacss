from logging.config import valid_ident
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, io, json, pickle
from functools import partial
from os.path import join

from tqdm import tqdm
import numpy as np
import optax
import treex as tx
import jax
import skimage.segmentation
from skimage.measure import regionprops
jnp = jax.numpy
import cloudpickle

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import lacss

from .data import *
from .util import *

import typer
app = typer.Typer(pretty_exceptions_enable=False)

def segment(img, scale=100.0, sigma=.8, min_size=20):
    img = img - np.mean(img, axis=(0,1))
    img = img / 2 / (np.linalg.norm(img, axis=(0,1))/512 + 1e-8) + .5
    im_mask = skimage.segmentation.felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
    return im_mask

def get_mask(label, mask_pred, loc=None):
    label = np.asarray(label).astype(int) + 1 # label 0 is valid
    mask_pred = np.asarray(mask_pred).astype(bool)
    if loc is not None:
        keep = label[tuple(np.round(loc).astype(int).transpose())].tolist()
    else:
        keep = []
    for r in regionprops(label):
        y0, x0, y1, x1 = r['bbox']
        patch = mask_pred[y0:y1, x0:x1]
        if r['area'] < np.count_nonzero(r['image'] & patch) * 2:
            keep.append(r['label'])
    return np.isin(label, keep)

def _to_mask(img, pred):
    label = jnp.zeros(img.shape[:-1])
    label = label.at[pred['instance_yc'], pred['instance_xc']].max(pred['instance_output'])
    return label

def get_mask_pred(model, img, loc):
    pred = predict(model.train(), img, loc)
    mask = jax.vmap(_to_mask)(img, pred)
    mask = mask > 0.5
    return mask

def train_data_gen_fn(teacher, data_path):
    X = np.load(join(data_path, 'train', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'train', 'y.npy'), mmap_mode='r+')
    L = np.load(join(data_path, 'train', 'S.npy'), mmap_mode='r+')
    for x, y, l in zip(X, Y, L):
        img = x.astype('float32')
        y = y[..., 0]
        locs = [prop['centroid'] for prop in regionprops(y)]

        mask_pred = get_mask_pred(
            teacher, 
            jnp.expand_dims(jnp.asarray(img), 0), 
            jnp.expand_dims(jnp.asarray(pad_to(locs)), 0),
        )
        binary_mask = get_mask(l, mask_pred[0], locs)

        data = dict(
            image = img,
            locations =  locs,
            binary_mask = binary_mask,
        )
        data = lacss.data.parse_train_data_func(data, size_jitter=[0.85, 1.15])

        if tf.shape(data['locations'])[0] > 0:

            data = dict(image = data['image'], gt_locations = pad_to(data['locations'].numpy())), dict(binary_mask=data['binary_mask'])        
            data = jax.tree_map(lambda v: jnp.asarray(v)[None,...], data)
        
            yield data

@app.command()
def run_training(
    checkpoint: str = '../log_tn_0/model_7',
    data_path: str = '../tissue_net',
    log_path: str = '.',
    n_epochs: int = 20,
    size_loss: float = 0.001
):
    print(f'Loading saved model {checkpoint}')
    with open(checkpoint, 'rb') as f:
        teacher = cloudpickle.load(f)
    teacher.detector._config_dict['test_max_output'] = 1024
    tr = lacss.train.Trainer(
        model = teacher.copy(),
        losses = [
            lacss.losses.LPNLoss(),
            lacss.losses.InstanceLoss(),
            lacss.losses.InstanceEdgeLoss(),
            SizeLoss(size_loss),
        ],
        optimizer = optax.adamw(0.001),
    )

    file_writer = tf.summary.create_file_writer(join(log_path, 'train'))
    preds = sample_prediction(data_path, teacher.train())
    with file_writer.as_default():
        tf.summary.image("inputs", sample_images(data_path), step=0, max_outputs=8)
        tf.summary.image("label_pred", to_label(preds), step=0, max_outputs=8)

    for epoch in range(1,n_epochs+1):
        print(f'epoch-{epoch}')
        ds = partial(train_data_gen_fn, teacher=teacher, data_path=data_path)
        t = tqdm(tr(ds))
        for log in t:
            # t.set_description(', '.join([f'{k}:{v:.4f}' for k,v in log.items()]))
            pass
        print(", ".join([f'{k}:{log[k]:.4f}' for k in log]))

        with open(join(log_path, f'model_{epoch}.pkl'), 'wb') as f:
            cloudpickle.dump(tr.model, f)

        teacher.merge(tr.model.parameters(), inplace=True)

        preds = sample_perdiction(datapath, teacher.train())
        label = to_label(preds)[...,None]
        label = label / label.max()
        with file_writer.as_default():
            tf.summary.image("label_pred", label, step=epoch, max_outputs=8)

if __name__ == "__main__":
    app()
