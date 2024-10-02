from __future__ import annotations

import warnings
from functools import partial, lru_cache
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from skimage.measure import regionprops
from skimage.transform import resize

import lacss.data.augment_ as augment
from lacss.data.utils import gf_batch, gf_cycle, image_standardization

SEG_SHAPE = (48, 48)
CACHE_SIZE = 16


class CtcVideo:
    def __init__(self, path, crop=None, scaling=1.0):
        self.path = Path(path)
        imgfns = {}
        for fn in sorted(list(self.path.glob("t*"))):
            frame = int(fn.stem[1:])
            imgfns[frame] = fn
        self.imgfns = imgfns
        self.crop = crop
        self.scaling = scaling

    def __getitem__(self, n):
        from lacss.data.generator import _to_masks as to_masks    

        if not isinstance(n, int):
            raise ValueError(f"int indexing only, got {n}")

        imgfn = self.imgfns[n]
        labeldir = imgfn.parent.parent / (imgfn.parent.stem + "_GT")

        img = imageio.imread(imgfn)
        if img.ndim == 2:
            img = img[..., None]
        data = {"image": img}

        trafn = labeldir / "TRA" / f"man_track{imgfn.stem[1:]}.tif"
        segfn = labeldir /"SEG"/f"man_seg{imgfn.stem[1:]}.tif"

        label = imageio.imread(trafn)
        locs = [rp.centroid for rp in regionprops(label)]
        data['centroids'] = np.array(locs)

        if segfn.exists():
            label = imageio.imread(segfn)
            if label.max() != len(data['centroids']):
                warnings.warn(f"TRA label and SEG label mismatch")
            else:
                locs = np.array([rp.centroid for rp in regionprops(label)]) + .5
                bboxes = np.array([rp.bbox for rp in regionprops(label)])
                masks = to_masks(label, bboxes, target_shape=SEG_SHAPE)
                data.update({"centroids":locs, "bboxes":bboxes, "cell_masks":masks})

        return data

    def __len__(self):
        return len(self.imgfns)


class CtmvVideo:
    def __init__(self, path):
        self.path = Path(path)
        imgdir = path/"img1"
        gt = np.loadtxt(path/"gt"/"gt.txt", delimiter=",", dtype=int)
        frames = gt[:, 0]
        all_bbox = gt[:, (3,2,5,4)]
        all_bbox = np.c_[all_bbox[:, :2], all_bbox[:, :2] + all_bbox[:,2:]]
        fns = sorted(list(imgdir.glob("*.jpg")))

        self.imgfns = fns
        self.frames = frames
        self.bbox = all_bbox

    def __getitem__(self, n):
        if not isinstance(n, int):
            raise ValueError(f"int indexing only, got {n}")
        imgfn = self.imgfns[n]

        img = imageio.imread(imgfn).squeeze()
        if img.ndim == 2:
            img = img[..., None]

        f = int(imgfn.stem)
        bboxes = self.bbox[self.frames == f]
        centroids = bboxes.reshape(-1, 2, 2).mean(1)

        data = dict(
            image = img,
            bboxes = bboxes,
            centroids = centroids,
        )

        return data

    def __len__(self):
        return len(self.imgfns)


@lru_cache(CACHE_SIZE)
def _prepare_video_data(video, frame, rescale=1.0, method=None):
    data = video[frame]
    data = augment.rescale(data, rescale=rescale)

    data['image'] = image_standardization(data['image'])

    # make sure img size is mutliple of 32
    imgsz = np.array(data['image'].shape[:-1], dtype=int)
    tgtsz = (imgsz - 1) // 32 * 32 + 32
    data = augment.pad_to_size(data, target_size=tuple(tgtsz))

    if method is not None:
        data['feature'] = method(data['image'])

    return data


def video_data_gen(all_movies, n_refs, method=None, padto=256):
    import random
    random.shuffle(all_movies)

    for mov, rescale in all_movies:
        for frm in range(len(mov)):
            ref_range = np.arange(frm - n_refs, frm + n_refs + 1)
            mask = (ref_range >= 0) & (ref_range < len(mov))
            ref_range = np.clip(ref_range, 0, len(mov)-1)

            data = _prepare_video_data(mov, frm, rescale, method)

            if n_refs > 0:
                ref_feature = []
                for ref_frm in ref_range:
                    ref_data = _prepare_video_data(mov, int(ref_frm), rescale, method)
                    ref_feature.append(ref_data['feature'])

                data['ref_feature'] = [np.stack(x) for x in zip(*ref_feature)]
                data['ref_mask'] = mask

            locs = np.asarray(data['centroids'])
            padding = (locs.shape[0]-1)//padto*padto+padto - locs.shape[0]
            data['centroids'] = np.pad(locs, [[0, padding],[0,0]], constant_values=-1)

            label = {}
            if "bboxes" in data:
                assert data['bboxes'].shape[0] == locs.shape[0]
                label['gt_bboxes'] = np.pad(data['bboxes'], [[0, padding],[0,0]], constant_values=-1)

            if "cell_masks" in data:
                assert data['cell_masks'].shape[0] == locs.shape[0]
                padding_list = [[0, padding]] + [[0,0]] * (data['cell_masks'].ndim - 1)
                label['gt_masks'] = np.pad(data['cell_masks'], padding_list)
            
            yield dict(
                image = data['image'],
                gt_locations = data['centroids'],
                video_refs = (data['ref_feature'], data['ref_mask']),
            ), label


ctc_catalog = {
    "BF-C2DL-HSC": 1,
    "BF-C2DL-MuSC": 0.8,
    "DIC-C2DH-HeLa": 35/140,
    "Fluo-C2DL-Huh7": 35 / 100,
    "Fluo-C2DL-MSC": 0.3,
    "Fluo-N2DH-SIM+": 35/ 53,
    "Fluo-N2DL-HeLa": 1,
    "PhC-C2DH-U373":35/ 90,
    "PhC-C2DL-PSC": 2.0,
}

ctmv_catalog = {
    'RK-13-run03': 88.63850971575397,
    '3T3-run03': 92.39983370576313,
    'MDBK-run01': 55.24779968615678,
    'APM-run03': 144.2420576478014,
    'MDOK-run05': 66.74546047117981,
    'U2O-S-run03': 103.8963597074468,
    '3T3-run07': 92.09185803757829,
    'LLC-MK2-run02a': 68.08149882903982,
    '3T3-run01': 121.01853747298988,
    'PL1Ut-run03': 65.06571451989673,
    'CV-1-run01': 119.37613122171946,
    'LLC-MK2-run01': 71.07403327090448,
    'OK-run07': 60.859908391668824,
    'CRE-BAG2-run03': 69.75049662296385,
    'BPAE-run07': 106.37924261952986,
    'MDBK-run05': 55.38616208622355,
    'A-10-run07': 70.03530885035255,
    'LLC-MK2-run05': 78.11411072608591,
    'MDBK-run03': 65.40681377129303,
    '3T3-run09': 109.30246136233544,
    'A-10-run01': 96.0993506008808,
    'BPAE-run03': 78.10840071262163,
    'CV-1-run03': 154.4066107860193,
    'PL1Ut-run01': 87.28143884892086,
    'CRE-BAG2-run01': 86.57473172238092,
    'APM-run05': 100.5180271947695,
    'APM-run01': 148.91313711414213,
    'OK-run03': 63.5259126754176,
    'MDBK-run09': 65.11865110246433,
    'BPAE-run05': 159.880437394857,
    'MDOK-run01': 245.4039608685278,
    'PL1Ut-run05': 115.24266936299293,
    '3T3-run05': 81.53608104741606,
    'BPAE-run01': 23.433367708743713,
    'RK-13-run01': 113.41445707070707,
    'MDBK-run07': 49.83901360448413,
    'MDOK-run03': 84.36965494469209,
    'U2O-S-run05': 73.9294292977996,
    'MDOK-run07': 114.88890716223995,
    'MDOK-run09': 85.88692194821792,
    'OK-run05': 72.40255160253086,
    'A-10-run03': 113.02312400816142,
    'A-10-run05': 127.92925010963309,
    'LLC-MK2-run07': 72.7524931778715,
    'LLC-MK2-run03': 72.61298686968995,
    'OK-run01': 77.90241305890703,
    'A-549-run03': 93.2961380268148
}

def ctc_video_ds(datapath, n_refs=0, method=None):
    datapath = Path(datapath)
    all_movies = [
        [CtcVideo(datapath/name/'train'/'01'), scaling]
        for (name, scaling) in ctc_catalog.items()
    ]
    all_movies += [
        [CtcVideo(datapath/name/'train'/'02'), scaling]
        for name, scaling in ctc_catalog.items()
    ]

    yield from gf_cycle(video_data_gen)(all_movies, n_refs, method)


def ctc_video_ds_test(datapath, name, n_refs=0, method=None):
    datapath = Path(datapath)
    all_movies = [
        (CtcVideo(datapath/name/'train'/'01'), ctc_catalog[name]),
        (CtcVideo(datapath/name/'train'/'02'), ctc_catalog[name]),
    ]

    yield from video_data_gen(all_movies, n_refs, method)


def ctmv_video_ds(datapath, n_refs=0, method=None):
    import random

    datapath = Path(datapath)
    
    all_movies = [
        (CtmvVideo(datapath/name), 35/sz)
        for name, sz in ctmv_catalog.items()
    ]

    random.shuffle(all_movies)

    yield from gf_cycle(video_data_gen)(all_movies, n_refs, method)



@gf_batch(32)
def ctmv_frame_ds(datapath, padding=128):
    datapath = Path(datapath)
    
    all_movies = [
        [CtmvVideo(datapath/name), 35/sz]
        for name, sz in ctmv_catalog.items()
    ]

    def _format(data, padding, rescale):
        data['image'] = image_standardization(data['image'])

        data = augment.rescale(data, rescale=rescale)
        data = augment.random_crop_or_pad(data, target_size=(256,256))

        n_cells = data['centroids'].shape[0]
        if n_cells < 2 or n_cells > padding: 
            return None

        n_pad = (n_cells-1)//padding*padding+padding - n_cells
        data['centroids'] = np.pad(data['centroids'], [[0, n_pad],[0,0]], constant_values=-1)
        data['bboxes'] = np.pad(data['bboxes'], [[0, n_pad],[0,0]], constant_values=-1)


        return data

    frame = 0
    while True:
        reset_frame = True
        for mov, rescale in all_movies:
            if frame < len(mov):
                data = _format(mov[frame], padding=padding, rescale=rescale)
                if data is not None:
                    reset_frame = False
                    yield dict(
                        image = data['image'],
                        gt_locations = data['centroids'],
                    ), dict(
                        gt_bboxes = data['bboxes'],
                    )
        frame = 0 if reset_frame else frame+1
