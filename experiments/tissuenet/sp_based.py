import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
from os.path import join
import numpy as np
import tensorflow as tf
import jax
import cloudpickle
import json
import pickle
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.color import label2rgb
import skimage.segmentation
from skimage.measure import regionprops
import optax
from functools import partial
from tqdm import tqdm
import io

import treex as tx

jnp = jax.numpy

tf.config.set_visible_devices([], 'GPU')
# jax.config.update('jax_platform_name', 'cpu')

import lacss

import typer
app = typer.Typer(pretty_exceptions_enable=False)

def show_images(imgs, locs=None):
    fig, axs = plt.subplots(1, len(imgs), figsize=(8*len(imgs), 10))
    for k, img in enumerate(imgs):
        axs[k].imshow(img)
        axs[k].axis('off')
        if locs is not None:
            loc = np.round(locs[k]).astype(int)
            for p in loc:
                c = patches.Circle((p[1], p[0]))
                axs[k].add_patch(c)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def segment(img, scale=100.0, sigma=.8, min_size=20):
    img = img - np.mean(img, axis=(0,1))
    img = img / 2 / (np.linalg.norm(img, axis=(0,1))/512 + 1e-8) + .5
    im_mask = skimage.segmentation.felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
    return im_mask

def pad_to(x, multiple=256):
    x = np.asarray(x)
    s = x.shape[0]
    ns = ((s - 1) // multiple + 1) * multiple
    padding = ns - s
    return np.pad(x, [[0,padding],[0,0]], constant_values=-1.0)

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

@jax.jit
def predict(model, *inputs):
    return model(*inputs)

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

class SizeLoss(tx.Loss):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def call(self, preds):
        valid_locs = (preds['instance_yc'] >= 0) & (preds['instance_yc'] < 512) & (preds['instance_xc'] >= 0) & (preds['instance_xc'] < 512)
        areas = jnp.sum(preds['instance_output'], axis=(2,3), where=valid_locs, keepdims=True) / 96 / 96
        loss =  self.a / (jnp.sqrt(areas)+1e-8)
        loss = jnp.sum(loss, axis=1, where=preds['instance_mask']) / (jnp.count_nonzero(preds['instance_mask'], axis=(1,2,3)) + 1e-8)
        return loss

to_label = jax.vmap(partial(lacss.ops.patches_to_label, input_size=(256,256)))

def plot_figures(data_path, model):
    X = np.load(join(data_path, 'val', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'val', 'y.npy'), mmap_mode='r+')
    rand8 = np.asarray([1830, 1071, 1579, 1666,  535, 3098,  493,  3000])
    img = X[rand8]
    y = Y[rand8]
    preds = predict(model.eval(), img)
    labels = to_label(preds)

    ch = np.argmax(np.count_nonzero(y, axis=(1,2)), axis=-1)
    y = [y0[..., ch0] for y0, ch0 in zip(y, ch)]
    getloc = lambda y: [prop['centroid'] for prop in regionprops(y)]
    locs = [getloc(p) for p in y]
    sz = max([len(l) for l in locs])
    psz = (sz-1)//256*256 + 256
    locs = np.asarray([np.pad(l, [[0,psz-len(l)],[0,0]], constant_values=-1.) for l in locs])
    preds_train = predict(model.train(),img, locs)
    labels_train = to_label(preds_train)

    fig1 = show_images(labels)
    fig2 = show_images(labels_train)
    return fig1, fig2

def plot_inputs(data_path):
    X = np.load(join(data_path, 'val', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'val', 'y.npy'), mmap_mode='r+')
    rand8 = np.asarray([1830, 1071, 1579, 1666,  535, 3098,  493,  3000])
    img = X[rand8]
    imgs = jnp.pad(img, [[0,0],[0,0],[0,0],[0,1]])[...,::-1]
    return show_images(imgs)

@app.command()
def run_training(
    checkpoint: str = '../log_tn_0/model_7',
    data_path: str = '../tissue_net',
    log_path: str = '.',
    n_epochs: int = 20,
):
    print(f'Loading saved model {checkpoint}')
    with open(checkpoint, 'rb') as f:
        teacher = cloudpickle.load(f)
    teacher.detector._config_dict['test_max_output'] = 1024
    tr = lacss.train.Trainer(
        model = teacher.copy(),
        losses = [
            lacss.losses.DetectionLoss(),
            lacss.losses.LocalizationLoss(),
            lacss.losses.InstanceLoss(),
            lacss.losses.InstanceEdgeLoss(),
            SizeLoss(0.001),
        ],
        optimizer = optax.adamw(0.001),
    )

    file_writer = tf.summary.create_file_writer(join(log_path, 'train'))
    with file_writer.as_default():
        fig = plot_inputs(data_path)
        tf.summary.image("inputs", fig, step=0)

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

        fig1, fig2 = plot_figures(data_path, tr.model)
        with file_writer.as_default():
            tf.summary.image("predictions", fig1, step=epoch)
            tf.summary.image("trainings", fig2, step=epoch)

if __name__ == "__main__":
    app()
