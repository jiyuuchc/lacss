import numpy as np
import pandas as pd


def decode_patch(codes, h, w):
    # decode string to a array representing the mask
    # inputs:
    #   codes: string
    #   h: height of patch, int
    #   w: width of patch, int
    # outputs:
    #   image:  array of hxw
    return (np.array([*codes]) == "1").astype(int).reshape(h, w)


def extract_frame(df, frame_no, img_height=512, img_width=512):
    # extract a single frame, label all cells
    # inputs:
    #   df: dataframe
    #   frame_no: int
    #   image_height: int
    #   image_width: int
    #  outputs:
    #   image: 2d array of [image_height, image_width]
    df_1 = df.loc[df["frame"] == frame_no]
    y0s = np.array(df_1["bbox_1"]).astype(int)
    x0s = np.array(df_1["bbox_2"]).astype(int)
    heights = np.array(df_1["bbox_3"]).astype(int)
    widths = np.array(df_1["bbox_4"]).astype(int)
    patches = np.array(df_1["image"])

    image = np.zeros([img_height, img_width], dtype=int)
    for k, (p, h, w, y, x) in enumerate(zip(patches, heights, widths, y0s, x0s)):
        decoded = decode_patch(p, h, w)
        decoded = decoded * (k + 1)
        image[y : y + h, x : x + w] = np.maximum(decoded, image[y : y + h, x : x + w])
    return image


def _data_generator(imgs, model, nms=True):

    from tqdm import tqdm

    from ..ops import non_max_suppress_predictions
    from ..train.strategy import JIT
    from ..utils import format_predictions

    for frame, img in enumerate(tqdm(imgs)):
        inputs = dict(image=img)
        pred = model(inputs)
        if nms:
            mask = non_max_suppress_predictions(pred)
        else:
            mask = None
        results = format_predictions(pred, mask)

        for box, code, score, roi in zip(*results):
            yield frame + 1, box[0], box[1], box[2] - box[0], box[3] - box[1], roi[
                0
            ], roi[1], score, code.count("1"), code,


def generate_predictions(imgs, model, nms=True):
    """segmentation predictions of a movie
    Args:
        imgs: an iterator of images sorted by frames
        model: a segmentation predictor or trainer
        nms: bool whether to perform nms
    Returns:
        pandas dataframe
    """
    df = pd.DataFrame(
        _data_generator(imgs, model, nms),
        columns=[
            "frame",
            "bbox_1",
            "bbox_2",
            "bbox_3",
            "bbox_4",
            "y",
            "x",
            "score",
            "area",
            "image",
        ],
    )

    return df


def to_labels(df, image_heigh=512, image_width=512):
    """
    Args:
        df: segmentation predictions
        image_height: int
        image_wdith: int
    returns:
        an iterator of image labels
    """

    max_frame = int(df["frame"].max())
    for frame in range(1, max_frame + 1):
        label = extract_frame(
            df,
            frame,
        )
        yield label
