"""Jaccard similarity between two clusterings (Ben-Hur et al., 2002).

Each clustering is a membership vector. The Jaccard similarity is
    J = <C1, C2> / (<C1, C1> + <C2, C2> - <C1, C2>),
where C[i, j] = 1 if points i and j are in the same cluster and i != j.

Four implementations are provided so their speeds can be compared:
    similarity_naive_python    O(q^2), pure Python double loop
    similarity_blocked_numpy   O(q^2), vectorized NumPy, row blocks
    similarity_table_numpy     O(q + k1*k2), contingency table with NumPy
    similarity_table_numba     O(q + k1*k2), contingency table with Numba
`similarity` is the fastest one (Numba) and is the function to use.
"""

import numpy as np
from numba import njit


def _jaccard_from_counts(both, pairs_1, pairs_2):
    """Combine pair counts into a Jaccard value.

    Args:
        both: ordered pairs together in both clusterings.
        pairs_1: ordered pairs together in the first clustering.
        pairs_2: ordered pairs together in the second clustering.

    Returns:
        Jaccard similarity; 1.0 if neither clustering groups any pair.
    """
    denominator = pairs_1 + pairs_2 - both
    if denominator == 0:
        return 1.0
    return both / denominator


def _relabel(membership):
    """Return labels as int64 values 0..k-1 (a no-op if already in range)."""
    membership = np.asarray(membership)
    if membership.size == 0 or (
        np.issubdtype(membership.dtype, np.integer) and membership.min() >= 0
    ):
        return membership.astype(np.int64, copy=False)
    return np.unique(membership, return_inverse=True)[1].astype(np.int64)


def _check_inputs(membership_1, membership_2):
    """Check that the two membership vectors are 1-D and equal length."""
    if membership_1.ndim != 1 or membership_1.shape != membership_2.shape:
        raise ValueError("membership vectors must be 1-D and the same length")


# --- Version 1: pure Python, O(q^2) ---------------------------------------
def similarity_naive_python(membership_1, membership_2):
    """Jaccard similarity by looping over every ordered pair of points."""
    labels_1 = list(membership_1)
    labels_2 = list(membership_2)
    n_points = len(labels_1)
    both = pairs_1 = pairs_2 = 0
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            same_1 = labels_1[i] == labels_1[j]
            same_2 = labels_2[i] == labels_2[j]
            pairs_1 += same_1
            pairs_2 += same_2
            both += same_1 and same_2
    return _jaccard_from_counts(both, pairs_1, pairs_2)


# --- Version 2: vectorized NumPy in row blocks, O(q^2) --------------------
def similarity_blocked_numpy(membership_1, membership_2, block_size=500):
    """Jaccard similarity with NumPy, one block of rows at a time.

    Time is still O(q^2), but memory is O(block_size * q) instead of O(q^2).
    """
    membership_1 = np.asarray(membership_1)
    membership_2 = np.asarray(membership_2)
    _check_inputs(membership_1, membership_2)
    n_points = membership_1.size
    both = pairs_1 = pairs_2 = 0
    for start in range(0, n_points, block_size):
        stop = min(start + block_size, n_points)
        same_1 = membership_1[start:stop, None] == membership_1[None, :]
        same_2 = membership_2[start:stop, None] == membership_2[None, :]
        both += np.count_nonzero(same_1 & same_2)
        pairs_1 += np.count_nonzero(same_1)
        pairs_2 += np.count_nonzero(same_2)
    # Each point matches itself once in every comparison; remove those.
    both -= n_points
    pairs_1 -= n_points
    pairs_2 -= n_points
    return _jaccard_from_counts(both, pairs_1, pairs_2)


# --- Version 3: contingency table with NumPy, O(q + k1*k2) ----------------
def similarity_table_numpy(membership_1, membership_2):
    """Jaccard similarity from the k1 x k2 contingency table (NumPy)."""
    labels_1 = _relabel(membership_1)
    labels_2 = _relabel(membership_2)
    _check_inputs(labels_1, labels_2)
    n_clusters_1 = labels_1.max() + 1
    n_clusters_2 = labels_2.max() + 1
    table = np.bincount(
        labels_1 * n_clusters_2 + labels_2,
        minlength=n_clusters_1 * n_clusters_2,
    ).astype(np.int64)
    both = np.sum(table * (table - 1))
    sizes_1 = table.reshape(n_clusters_1, n_clusters_2).sum(axis=1)
    sizes_2 = table.reshape(n_clusters_1, n_clusters_2).sum(axis=0)
    pairs_1 = np.sum(sizes_1 * (sizes_1 - 1))
    pairs_2 = np.sum(sizes_2 * (sizes_2 - 1))
    return _jaccard_from_counts(int(both), int(pairs_1), int(pairs_2))


# --- Version 4: contingency table with Numba, O(q + k1*k2) ----------------
@njit(cache=True)
def _pair_counts_numba(labels_1, labels_2, n_clusters_1, n_clusters_2):
    """One pass over the data to build the table, then count pairs from it."""
    table = np.zeros((n_clusters_1, n_clusters_2), dtype=np.int64)
    for i in range(labels_1.size):
        table[labels_1[i], labels_2[i]] += 1
    both = 0
    pairs_1 = 0
    pairs_2 = 0
    sizes_2 = np.zeros(n_clusters_2, dtype=np.int64)
    for a in range(n_clusters_1):
        size_1 = 0
        for b in range(n_clusters_2):
            count = table[a, b]
            both += count * (count - 1)
            size_1 += count
            sizes_2[b] += count
        pairs_1 += size_1 * (size_1 - 1)
    for b in range(n_clusters_2):
        pairs_2 += sizes_2[b] * (sizes_2[b] - 1)
    return both, pairs_1, pairs_2


def similarity_table_numba(membership_1, membership_2):
    """Jaccard similarity from the contingency table (Numba, fastest)."""
    labels_1 = _relabel(membership_1)
    labels_2 = _relabel(membership_2)
    _check_inputs(labels_1, labels_2)
    if labels_1.size == 0:
        return 1.0
    both, pairs_1, pairs_2 = _pair_counts_numba(
        labels_1, labels_2, labels_1.max() + 1, labels_2.max() + 1
    )
    return _jaccard_from_counts(both, pairs_1, pairs_2)


def similarity(membership_1, membership_2):
    """Jaccard similarity between two membership vectors.

    Args:
        membership_1: 1-D array of cluster labels, one per point.
        membership_2: 1-D array of cluster labels for the same points.

    Returns:
        A float in [0, 1]; larger means the clusterings agree more.
    """
    return similarity_table_numba(membership_1, membership_2)
