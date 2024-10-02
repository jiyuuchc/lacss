from functools import wraps, partial
import numpy as np
import jax

def image_standardization(img):
    img = img - img.mean()
    axis = tuple(range(img.ndim-1))
    img = img / img.std(axis=axis, keepdims=True)

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 2:
        img = np.c_[img, np.zeros_like(img[..., :1])]

    assert img.shape[-1] == 3

    return img


def gf_cycle(gf):
    @wraps(gf)
    def _wrapped(*args, **kwargs):
        while True:
            yield from gf(*args, **kwargs)

    return _wrapped


def gf_batch_(gf, *, batch_size):
    @wraps(gf)
    def _wrapped(*args, **kwargs):
        stopping = False
        it = gf(*args, **kwargs)
        while not stopping:
            data = []
            for _ in range(batch_size):
                try:
                    data.append(next(it))
                except StopIteration:
                    stopping = True
                    break
            data = jax.tree_util.tree_map(lambda *x: np.stack(x), *data)

            yield data

    assert isinstance(batch_size, int)

    return _wrapped


def gf_batch(batch_size):
    return partial(gf_batch_, batch_size=batch_size)