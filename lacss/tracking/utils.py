import jax
import jax.numpy as jnp
import numpy as np


def select_sample(history):
    """a naive sample selection logic
    find the mean cells-per-frame, and return the sample that has the most similar values
    """

    cpf = np.array(
        [np.count_nonzero(s["selected"], axis=1) for s in history["samples"]]
    )
    m_cpf = cpf.mean(axis=1)
    mse = ((cpf - m_cpf[:, None]) ** 2).sum(axis=0)

    idx = np.argmin(mse)
    # plt.plot(cpf[:,idx])

    track = jax.tree_util.tree_map(lambda v: v[idx], history["samples"])
    dets = history["detections"]

    return track, dets


def update_with_sample(df, track, detections, *, n_frames=-1):

    df_tracking = df.copy()  # df still has the index column

    if n_frames <= 0:
        n_frames = int(df["frame"].max())

    df_tracking["c0"] = -1
    df_tracking["c1"] = -1
    for k in range(n_frames - 1):
        sel = track[k]["selected"].astype(bool)
        current_ids = detections[k]["id"][sel]
        next_ids = detections[k]["id_next"]
        links, links_div = np.moveaxis(track[k]["next"][sel], -1, 0)
        df_tracking.loc[current_ids, "c0"] = np.where(links < 0, links, next_ids[links])
        df_tracking.loc[current_ids, "c1"] = np.where(
            links_div < 0, links_div, next_ids[links_div]
        )

    # label cp
    df_tracking["cp"] = -1
    has_child_1 = df_tracking["c0"] > 0
    child_1_ids = df_tracking.loc[has_child_1]["c0"]
    df_tracking.loc[child_1_ids, "cp"] = df_tracking.loc[has_child_1][
        "index"
    ].to_numpy()
    has_child_2 = df_tracking["c1"] > 0
    child_2_ids = df_tracking.loc[has_child_2]["c1"]
    df_tracking.loc[child_2_ids, "cp"] = df_tracking.loc[has_child_2][
        "index"
    ].to_numpy()

    # remove orphan
    is_orphan = (
        (df_tracking["cp"] == -1)
        & (df_tracking["c0"] == -1)
        & (df_tracking["c1"] == -1)
    )
    df_tracking = df_tracking.loc[~is_orphan]

    # don't need the index column any more
    df_tracking = df_tracking.rename({"index": "cell_id"}, axis=1)

    # recursive tree iteration
    def id_cells(df, root=0):
        max_id = 1
        df["index"] = -1
        df["parent_idx"] = -1
        df["child_idx_1"] = -1
        df["child_idx_2"] = -1

        def _label(root, cur_id, parent):
            nonlocal max_id
            rows = [root]
            cur_row = root
            while df.loc[cur_row, "c0"] != -1 and df.loc[cur_row, "c1"] == -1:
                cur_row = df.loc[cur_row, "c0"]
                rows.append(cur_row)
            df.loc[rows, "index"] = cur_id
            df.loc[rows, "parent_idx"] = parent
            if df.loc[cur_row, "c0"] < 0:  # end of lineage
                return
            else:
                max_id += 1
                df.loc[rows, "child_idx_1"] = max_id
                _label(df.loc[cur_row, "c0"], max_id, cur_id)

                new_id_2 = max_id + 1
                df.loc[rows, "child_idx_2"] = max_id
                _label(df.loc[cur_row, "c1"], max_id, cur_id)

        _label(root, max_id, -1)

    id_cells(df_tracking)

    # df_tracking = df_tracking.drop(['c0', 'c1', 'cp'], axis=1)
    df_tracking = df_tracking.set_index("index")

    # the index should now reflect cell_id
    return df_tracking
