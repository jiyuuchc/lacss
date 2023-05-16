import jax
import jax.numpy as jnp
import numpy as np


def select_sample(chains):
    """a naive sample selection logic
    find the mean tracked_vector, and return the sample that has the most similar values
    """

    a = chains["tracked"] - chains["tracked"].mean(axis=1, keepdims=True)
    a = (a**2).sum(axis=(0, 2))

    return np.argmin(a)


def update_with_sample(df, chains, idx):

    tracked = chains["tracked"][:, idx, :]
    parent = chains["parent"][:, idx, :]
    age = chains["age"][:, idx, :]

    df_tracking = df.copy()
    df_tracking["parent"] = -1
    df_tracking["tracked"] = False
    df_tracking["age"] = 0

    n_frame = len(tracked)
    ndets = [
        len(df_tracking.loc[df_tracking["frame"] == k + 1]) for k in range(n_frame)
    ]
    first_ids = [
        df_tracking.loc[df_tracking["frame"] == k + 1, "index"].iloc[0]
        for k in range(n_frame)
    ]

    def pad_or_truncate(a, length, constant_values=0):
        if len(a) >= length:
            return a[:length]
        else:
            return np.pad(a, [0, length - len(a)], constant_values=constant_values)

    for k in range(n_frame, 0, -1):
        t = pad_or_truncate(
            tracked[k - 1],
            ndets[k - 1],
        )
        df_tracking.loc[df_tracking["frame"] == k, "tracked"] = t

        p = pad_or_truncate(parent[k - 1], ndets[k - 1], constant_values=-1)
        p = np.where(p >= 0, p + first_ids[k - 2], p)
        df_tracking.loc[df_tracking["frame"] == k, "parent"] = p

        a = pad_or_truncate(
            age[k - 1],
            ndets[k - 1],
        )
        df_tracking.loc[df_tracking["frame"] == k, "age"] = a

    df_tracking = df_tracking.loc[df_tracking["tracked"]].drop("tracked", axis=1)

    df_tracking["cell_id"] = -1
    cell_id = 0

    for id0 in df_tracking.index[::-1]:
        if df_tracking.loc[id0, "cell_id"] != -1:
            continue

        id1 = id0
        while id1 != -1:
            df_tracking.loc[id1, "cell_id"] = cell_id
            if df_tracking.loc[id1, "age"] != 0:
                id1 = df_tracking.loc[id1, "parent"]
            else:
                break
        cell_id += 1

    df_tracking["cell_id"] = df_tracking["cell_id"].max() - df_tracking["cell_id"]

    df_tracking["child_1"] = -1
    df_tracking["child_2"] = -1
    df_tracking["parent_cell_id"] = -1

    for cell_id in range(df_tracking["cell_id"].max() + 1):

        parent_id = df_tracking.loc[
            df_tracking["cell_id"] == cell_id, "parent"
        ].to_numpy()[0]

        if parent_id != -1:
            parent_cell_id = df_tracking.loc[parent_id, "cell_id"]

            df_tracking.loc[
                df_tracking["cell_id"] == cell_id, "parent_cell_id"
            ] = parent_cell_id

            if (
                df_tracking.loc[
                    df_tracking["cell_id"] == parent_cell_id, "child_1"
                ].iloc[0]
                == -1
            ):
                df_tracking.loc[
                    df_tracking["cell_id"] == parent_cell_id, "child_1"
                ] = cell_id
            else:
                df_tracking.loc[
                    df_tracking["cell_id"] == parent_cell_id, "child_2"
                ] = cell_id

    df_tracking = df_tracking.rename(columns={"parent": "parent_id"})
    df_tracking = df_tracking.rename(columns={"parent_cell_id": "parent"})

    return df_tracking


def _box_intersect(gt, pred):
    start_frame = max(gt["frame"].min(), pred["frame"].min())
    end_frame = min(gt["frame"].max(), pred["frame"].max())

    gt = gt.loc[(gt["frame"] >= start_frame) & (gt["frame"] <= end_frame)]
    pred = pred.loc[(pred["frame"] >= start_frame) & (pred["frame"] <= end_frame)]

    gt_y0, gt_x0, gt_h, gt_w = (
        gt[["bbox_y", "bbox_x", "bbox_h", "bbox_w"]].to_numpy().transpose()
    )
    gt_y1 = gt_y0 + gt_h
    gt_x1 = gt_x0 + gt_w

    pred_y0, pred_x0, pred_h, pred_w = (
        pred[["bbox_y", "bbox_x", "bbox_h", "bbox_w"]].to_numpy().transpose()
    )
    pred_y1 = pred_y0 + pred_h
    pred_x1 = pred_x0 + pred_w

    y_min_max = np.minimum(gt_y1, pred_y1)
    y_max_min = np.maximum(gt_y0, pred_y0)
    x_min_max = np.minimum(gt_x1, pred_x1)
    x_max_min = np.maximum(gt_x0, pred_x0)

    intersect_heights = y_min_max - y_max_min
    intersect_widths = x_min_max - x_max_min
    intersect_heights = np.maximum(0, intersect_heights)
    intersect_widths = np.maximum(0, intersect_widths)

    return intersect_heights * intersect_widths


def broi(gt, pred):
    gt_area = (gt["bbox_h"] * gt["bbox_w"]).sum()
    pred_area = (pred["bbox_h"] * pred["bbox_w"]).sum()
    intersects = _box_intersect(gt, pred).sum()
    roi = intersects / (gt_area + pred_area - intersects + 1e-8)
    return roi


def proi(gt, pred, g=30):
    total_len = len(gt) + len(pred)

    start_frame = max(gt["frame"].min(), pred["frame"].min())
    end_frame = min(gt["frame"].max(), pred["frame"].max())

    gt = gt.loc[(gt["frame"] >= start_frame) & (gt["frame"] <= end_frame)]
    pred = pred.loc[(pred["frame"] >= start_frame) & (pred["frame"] <= end_frame)]

    yofs = np.minimum(np.abs(gt["y"].to_numpy() - pred["y"].to_numpy()) / g, 1.0)
    xofs = np.minimum(np.abs(gt["x"].to_numpy() - pred["x"].to_numpy()) / g, 1.0)
    its = ((1 - yofs) * (1 - xofs)).sum()
    proi = its / (total_len - its + 1e-8)
    return proi


def filter_ious(ious):

    ious_out = np.zeros_like(ious)
    ious = ious.copy()

    for _ in range(len(ious)):
        row, col = np.unravel_index(ious.argmax(), ious.shape)
        ious_out[row, col] = ious[row, col]
        ious[row, :] = -1
        ious[:, col] = -1

    return ious_out
