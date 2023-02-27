import io
from functools import partial
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
import treex as tx
import jax
jnp = jax.numpy
EPS = jnp.finfo("float32").eps

import tensorflow as tf

import lacss

_RAND_8 = np.asarray([1830, 1071, 1579, 1666,  535, 3098,  493,  3000])

to_label = jax.vmap(partial(lacss.ops.patches_to_label, input_size=(256,256)))

@jax.jit
def predict(model, *inputs):
    return model(*inputs)

def show_images(imgs, locs=None):
    outs = []
    for img in imgs:
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4).numpy()
        outs.append(image)
    return np.asarray(outs)

def sample_prediction(data_path, model, rand8 = _RAND_8):
    X = np.load(join(data_path, 'val', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'val', 'y.npy'), mmap_mode='r+')
    T = np.load(join(data_path, 'val', 'tissue_list.npy'), mmap_mode='r+')
    img = X[rand8]
    y = Y[rand8]

    ch = np.argmax(np.count_nonzero(y, axis=(1,2)), axis=-1)
    y = [y0[..., ch0] for y0, ch0 in zip(y, ch)]
    getloc = lambda y: [prop['centroid'] for prop in regionprops(y)]
    locs = [getloc(p) for p in y]
    sz = max([len(l) for l in locs])
    psz = (sz-1)//256*256 + 256
    locs = np.asarray([np.pad(l, [[0,psz-len(l)],[0,0]], constant_values=-1.) for l in locs])

    preds = predict(model, img, locs)

    if preds['fg_pred'].shape[-1] > 0:
        _tissue_types={
            'breast': 0, 'gi': 1, 'immune': 2, 'lung': 3, 'pancreas': 4, 'skin':5,
        }
        ts = [_tissue_types[T[k]] for k in rand8]
        preds['fg_pred'] = jnp.asarray([x[...,t:t+1] for x, t in zip(preds['fg_pred'], ts)])

    # fig1 = show_images(labels)
    # fig1 = show_images(preds_train['fg_pred'])
    # fig2 = show_images(to_label(preds_train))

    return preds

def sample_images(data_path, rand8 = _RAND_8):
    X = np.load(join(data_path, 'val', 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'val', 'y.npy'), mmap_mode='r+')
    img = X[rand8]
    img = np.concatenate([img, np.zeros_like(img[...,:1])], axis=-1)

    return img[...,::-1]

class SizeLoss(tx.Loss):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def call(self, preds):
        valid_locs = (preds['instance_yc'] >= 0) & (preds['instance_yc'] < 512) & (preds['instance_xc'] >= 0) & (preds['instance_xc'] < 512)
        areas = jnp.sum(preds['instance_output'], axis=(2,3), where=valid_locs, keepdims=True) / 96 / 96
        areas = jnp.clip(areas, EPS, 1.)
        loss =  self.a * jnp.clip(jax.lax.rsqrt(areas), 0., 1.e8)
        loss = jnp.sum(loss, axis=1, where=preds['instance_mask']) / (jnp.count_nonzero(preds['instance_mask'], axis=(1,2,3)) + 1e-8)
        return loss
