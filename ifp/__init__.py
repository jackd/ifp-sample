"""
Example usage:

```python
import numpy as np
from scipy.spatial import cKDTree
from ifp import ifp_sample

coords = np.random.uniform(size=(1024, 3))
dists, indices = cKDTree(coords).query(coords, 8)
sampled_indices = ifp_sample(dists, indices, num_out=512)
```
"""
import numpy as np
import heapq
# pylint: disable=no-name-in-module
from ifp._ifp import ifp_sample_heap_unchecked
from ifp._ifp import ifp_sample_heap_ragged_unchecked

from ifp._rej import rejection_sample_unchecked
from ifp._rej import rejection_sample_ragged_unchecked

from ifp._hybrid import ifp_sample_hybrid_prealloc
# pylint: enable=no-name-in-module


def _validate_input(value, name, rank=None, dtype=None):
    if rank is not None:
        if len(value.shape) != rank:
            raise ValueError('{} must have rank {} but has shape {}'.format(
                name, rank, value.shape))
    if dtype is not None:
        if value.dtype != dtype:
            raise ValueError('{} must have dtype {} but has {}'.format(
                name, dtype, value.dtype))


def validated_ifp_sample_args(dists, indices, out_size):
    dists = np.asanyarray(dists, np.float32)
    indices = np.asanyarray(indices, np.uint32)
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
    if np.any(np.logical_and(indices >= in_size, np.isfinite(dists))):
        raise ValueError(
            'indices must all be less than in_size but got max value {} >= {}'.
            format(np.max(indices), in_size))
    return dists, indices


def ifp_sample_hybrid(dists, indices, out_size):
    """
    Iterative farthest point sampling with pre-calculated distances.

    This implementation performs rejection sampling that also records minimum
    distances and then potentially finishes off with a priority queue-based
    implementation.

    Args:
        dists: [in_size, K] floats
        indices: [in_size, K] ints in [0, in_size)
        out_size: int, (maximum) number of outputs

    Returns:
        indices: [out_size] uint32 indices of sampled points.
    """
    dists, indices = validated_ifp_sample_args(dists, indices, out_size)
    in_size = indices.shape[0]
    out = np.empty((out_size,), dtype=np.uint32)
    consumed = np.zeros((in_size,), dtype=np.uint8)
    min_dists = np.full((in_size,), np.inf, dtype=np.float32)

    ifp_sample_hybrid_prealloc(dists.astype(np.float32),
                               indices.astype(np.uint32), out_size, min_dists,
                               consumed, out)
    return out


def ifp_sample_pq(dists, indices, out_size):
    """
    Iterative farthest point sampling with pre-calculated distances.

    This implementation uses a priority queue. ifp_sample_hybrid may give
    better performance for small neighborhoods.

    Args:
        dists: [in_size, K] floats
        indices: [in_size, K] ints in [0, in_size)
        out_size: int, (maximum) number of outputs

    Returns:
        indices: [out_size] uint32 indices of sampled points.
    """
    dists, indices = validated_ifp_sample_args(dists, indices, out_size)
    return ifp_sample_heap_unchecked(dists, indices, out_size=out_size)


ifp_sample = ifp_sample_hybrid


def ifp_sample_ragged(dists, indices, row_splits, out_size):
    dists = np.asanyarray(dists, np.float32)
    indices = np.asanyarray(indices, np.uint32)
    row_splits = np.asanyarray(row_splits, np.uint32)
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
    if np.any(np.logical_and(indices >= in_size, np.isfinite(dists))):
        raise ValueError(
            'indices should all be under in_size, but {} >= {}'.format(
                np.max(indices), in_size))

    if out_size > in_size:
        raise ValueError(
            'out_size cannot be greater than input size, but {} > {}'.format(
                out_size, in_size))

    return ifp_sample_heap_ragged_unchecked(dists, indices, row_splits,
                                            out_size)


def rejection_sample_prealloc(indices: np.ndarray, consumed: np.ndarray,
                              out: np.ndarray):
    _validate_input(indices, 'indices', 2, np.uint32)
    _validate_input(consumed, 'consumed', 1, np.uint8)
    _validate_input(out, 'out', 1, np.uint32)

    return rejection_sample_unchecked(indices, consumed, out)


def rejection_sample(indices: np.ndarray, max_size: int):
    _validate_input(indices, 'indices', 2)
    out = np.empty((max_size,), dtype=np.uint32)

    in_size = indices.shape[0]
    consumed = np.zeros((in_size,), dtype=np.uint8)

    n, _ = rejection_sample_unchecked(indices.astype(np.uint32), consumed, out)
    return out[:n]


def rejection_sample_ragged_prealloc(indices: np.ndarray,
                                     row_splits: np.ndarray,
                                     consumed: np.ndarray, out: np.ndarray):
    _validate_input(indices, 'indices', 1, np.uint32)
    _validate_input(indices, 'row_splits', 1, np.uint32)
    _validate_input(consumed, 'consumed', 1, np.uint8)
    _validate_input(out, 'out', 1, np.uint32)

    return rejection_sample_ragged_unchecked(indices, consumed, out)


def rejection_sample_ragged(indices: np.ndarray, row_splits: np.ndarray,
                            max_size: int):
    _validate_input(indices, 'indices', 1)
    _validate_input(indices, 'row_splits', 1)
    out = np.empty((max_size,), dtype=np.uint32)

    in_size = row_splits.size - 1
    consumed = np.zeros((in_size,), dtype=np.uint8)

    n, _ = rejection_sample_ragged_unchecked(
        np.asanyarray(indices, np.uint32), np.asanyarray(row_splits, np.uint32),
        consumed, out)
    return out[:n]
