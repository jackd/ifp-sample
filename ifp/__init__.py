from ifp._ifp import ifp_sample_heap_unchecked  # pylint: disable=no-name-in-module
from ifp._ifp import ifp_sample_heap_ragged_unchecked  # pylint: disable=no-name-in-module
import numpy as np


def ifp_sample(dists, indices, out_size):
    """
    Iterative farthest point sampling with pre-calculated distances using heapq.

    Example usage:

    ```python
    import numpy as np
    from scipy.spatial import cKDTree
    from ifp import ifp_sample

    coords = np.random.uniform(size=(1024, 3))
    dists, indices = cKDTree(coords).query(coords, 8)
    sampled_indices = ifp_sample(dists, indices, num_out=512)
    ```

    Args:
        dists: [in_size, K] floats
        indices: [in_size, K] ints in [0, in_size)
        out_size: int, (maximum) number of outputs

    Returns:
        indices: [out_size] uint32 indices of sampled points.
    """
    dists = dists.astype(np.float32)
    indices = indices.astype(np.uint32)
    in_size = indices.shape[0]
    if out_size > in_size:
        raise ValueError(
            'out_size cannot be greater than input size, but {} > {}'.format(
                out_size, in_size))
    if len(indices.shape) != 2:
        raise ValueError('indices must be rank 2, got shape {}'.format(
            indices.shape))
    if dists.shape != indices.shape:
        raise ValueError('indices.shape must be the same as dists.shape, '
                         'but {} != {}'.format(indices.shape, dists.shape))
    if np.any(indices >= in_size):
        raise ValueError(
            'indices must all be less than in_size but got max value {} >= {}'.
            format(np.max(indices), in_size))
    return ifp_sample_heap_unchecked(dists, indices, out_size=out_size)


def ifp_sample_ragged(dists, indices, row_splits, out_size):
    dists = dists.astype(np.float32)
    indices = indices.astype(np.uint32)
    row_splits = row_splits.astype(np.uint32)
    if dists.shape != indices.shape:
        raise ValueError(
            'dists must have same shape as indices, got {} and {}'.format(
                dists.shape, indices.shape))

    total = row_splits[-1]
    if dists.shape != (total,):
        raise ValueError(
            'dists/indices.shape should be (row_splits[-1],), but {} != {}'.
            format(dists.shape, (total,)))

    in_size = row_splits.size - 1
    if np.any(indices >= in_size):
        raise ValueError(
            'indices should all be under in_size, but {} >= {}'.format(
                np.max(indices), in_size))

    if out_size > in_size:
        raise ValueError(
            'out_size cannot be greater than input size, but {} > {}'.format(
                out_size, in_size))

    return ifp_sample_heap_ragged_unchecked(dists, indices, row_splits,
                                            out_size)
