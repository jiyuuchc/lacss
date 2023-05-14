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
