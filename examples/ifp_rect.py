from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba
import heapq
from scipy.spatial import cKDTree as KDTree  # pylint: disable=no-name-in-module
from timeit import timeit
import functools

from ifp import ifp_sample_pq
from ifp import ifp_sample_hybrid

from benchmark_util import run_benchmarks

np.random.seed(123)

k = 32
in_size = 1024
out_size = 512

num_iters = 100
burn_iters = 50


def ifp_sample_base(indices, dists, out_size):
    """
    Args:
        indices: [in_size, K] ints in [0, in_size)
        dists: [in_size, K] floats
        out_size: int, (maximum) number of outputs

    Returns:
        indices: [out_size] int indices of sampled points.
    """
    indices = indices.astype(np.uint32)
    dists = dists.astype(np.float32)
    # remove self-dists / make dists negative
    dists = -dists[:, 1:]
    indices = indices[:, 1:]
    out = -np.ones((out_size,), dtype=np.int64)
    in_size, _ = indices.shape
    if out_size > in_size:
        raise ValueError('Requires out_size <= in_size, but {} > {}'.format(
            out_size, in_size))
    i = np.arange(in_size, dtype=np.uint32)
    heap_dists = np.empty((in_size,), dtype=np.float32)
    heap_dists[:] = -np.inf
    heap = list(zip(heap_dists, i))
    del i
    count = 0

    while heap:
        dist, index = heapq.heappop(heap)
        if dist != heap_dists[index]:
            continue
        out[count] = index
        count += 1
        if count >= out_size:
            break
        di = dists[index]
        ii = indices[index]
        new_heap_dists = heap_dists[ii]
        needs_updating = di > new_heap_dists
        ii = ii[needs_updating]
        di = di[needs_updating]
        heap_dists[ii] = di
        heap_dists[index] = 0
        for item in zip(di, ii):
            heapq.heappush(heap, item)
    else:
        raise RuntimeError('Should have broken...')
    return out


@numba.njit()
def _ifp_sample_numba(indices, dists, heap_dists, i, out):
    """
    Args:
        indices: [in_size, K] ints in [0, in_size)
        dists: [in_size, K] floats
        out_size: int, (maximum) number of outputs

    Returns:
        indices: [out_size] int indices of sampled points.
    """
    heap = list(zip(heap_dists, i))
    count = 0
    out_size = out.shape[0]

    while heap:
        dist, index = heapq.heappop(heap)
        if dist != heap_dists[index]:
            continue
        out[count] = index
        count += 1
        if count >= out_size:
            break
        di = dists[index]
        ii = indices[index]
        new_heap_dists = heap_dists[ii]
        needs_updating = di > new_heap_dists
        ii = ii[needs_updating]
        di = di[needs_updating]
        heap_dists[ii] = di
        heap_dists[index] = 0
        for item in zip(di, ii):
            heapq.heappush(heap, item)
    else:
        raise RuntimeError('Should have broken...')
    return out


def ifp_sample_numba(indices, dists, out_size):
    indices = indices.astype(np.uint32)
    dists = dists.astype(np.float32)
    # remove self-dists / make dists negative
    dists = -dists[:, 1:]
    indices = indices[:, 1:]
    out = -np.ones((out_size,), dtype=np.int64)
    in_size, _ = indices.shape
    if out_size > in_size:
        raise ValueError('Requires out_size <= in_size, but {} > {}'.format(
            out_size, in_size))
    i = np.arange(in_size, dtype=np.uint32)
    heap_dists = np.full((in_size,), -np.inf, dtype=np.float32)
    # heap = list(zip(heap_dists, i))
    return _ifp_sample_numba(indices, dists, heap_dists, i, out)


np.random.seed(123)
x = np.random.uniform(size=(in_size, 2)).astype(dtype=np.float32)
tree = KDTree(x)
dists, indices = tree.query(x, k)

kwargs = dict(indices=indices, dists=dists, out_size=out_size)

fn_pq = functools.partial(ifp_sample_pq, **kwargs)
fn_hybrid = functools.partial(ifp_sample_hybrid, **kwargs)
fn_base = functools.partial(ifp_sample_base, **kwargs)
fn_numba = functools.partial(ifp_sample_numba, **kwargs)

run_benchmarks(burn_iters, num_iters, ('pq', fn_pq), ('hybrid', fn_hybrid),
               ('base', fn_base), ('numba', fn_numba))

pq_indices = fn_pq()
hybrid_indices = fn_hybrid()
base_indices = fn_base()
numba_indices = fn_numba()
# print(np.stack((base_indices, pq_indices, hybrid_indices), axis=1))
# print(['base', 'pq', 'hybrid'])
print('All same (pq vs base)    : {}'.format(
    np.all(pq_indices == base_indices)))
print('All same (hybrid vs base): {}'.format(
    np.all(hybrid_indices == base_indices)))
print('All same (pq vs hybrid)  : {}'.format(
    np.all(pq_indices == hybrid_indices)))
print('All same (numba vs base)  : {}'.format(
    np.all(numba_indices == base_indices)))

import matplotlib.pyplot as plt
x_out = x[pq_indices]
plt.scatter(*x.T, c='k')
plt.scatter(*(x_out.T), c='r')
plt.show()
