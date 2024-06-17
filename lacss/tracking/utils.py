import numpy as np

def linear_assignment(cost_matrix, thresh):
    """ perform linear assignment 
    Args:
        cost_matrix: M x N cost matrix
        thresh: max cost for valid assignment

    Returns:
        matches: list of [idx_a, idx_b] pair indicating matches
        unmatched_a: all unmatched idx_a
        unmatched_b: all unmatched idx_b
    """
    import lap

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    matches = []
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    matches = np.asarray(matches)

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]

    return matches, unmatched_a, unmatched_b
