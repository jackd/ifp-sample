"""Followed guide at https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial ."""

cimport cython
import numpy as np
import heapq


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def ifp_sample_heap_unchecked(
        float[:, ::1] dists, unsigned int[:, ::1] indices, Py_ssize_t out_size):
    """
    Args:
        dists: [in_size, K] float32
        indices: [in_size, K] uint32 in [0, in_size)
        out_size: int, (maximum) number of outputs

    Returns:
        indices: [out_size] uint32 indices of sampled points.
    """
    cdef Py_ssize_t in_size = indices.shape[0]
    cdef Py_ssize_t K = indices.shape[1]
    cdef int count = 0
    cdef int index
    cdef float dist
    cdef float di
    cdef float old_di
    cdef int i
    cdef int k

    cdef float neg_inf = -np.inf

    out = np.empty((out_size,), dtype=np.intc)
    heap_dists = np.empty((in_size,), dtype=np.float32)

    cdef float[::1] heap_dists_view = heap_dists
    for i in range(in_size):
        heap_dists_view[i] = neg_inf
    cdef int[::1] out_view = out

    heap = [(neg_inf, float(i)) for i in range(in_size)]

    for count in range(out_size):
        dist, index = heapq.heappop(heap)
        while dist != heap_dists_view[index]:
            dist, index = heapq.heappop(heap)
        out_view[count] = index
        heap_dists_view[index] = 0
        for k in range(1, K):
            i = indices[index, k]
            di = -dists[index, k]

            old_di = heap_dists_view[i]
            if di > old_di:
                heap_dists_view[i] = di
                heapq.heappush(heap, (di, i))
    return out


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def ifp_sample_heap_ragged_unchecked(
        float[::1] dists, unsigned int[::1] indices,
        unsigned int[::1] row_splits,  Py_ssize_t out_size):
    """
    Args:
        dists: [E] float32
        indices: [E] uint32 in [0, in_size)
        row_splits: [in_size+1] uint32 giving start/end indices of each
            ragged entry of dists/indices.
        out_size: int, (maximum) number of outputs

    Returns:
        indices: [out_size] uint32 indices of sampled points.
    """
    cdef Py_ssize_t in_size = row_splits.shape[0] - 1
    cdef int count = 0
    cdef int index
    cdef float dist
    cdef float di
    cdef float old_di
    cdef int i
    cdef int k

    cdef float neg_inf = -np.inf

    out = np.empty((out_size,), dtype=np.intc)
    heap_dists = np.empty((in_size,), dtype=np.float32)

    cdef float[::1] heap_dists_view = heap_dists
    for i in range(in_size):
        heap_dists_view[i] = neg_inf
    cdef int[::1] out_view = out

    heap = [(neg_inf, float(i)) for i in range(in_size)]

    for count in range(out_size):
        dist, index = heapq.heappop(heap)
        while dist != heap_dists_view[index]:
            dist, index = heapq.heappop(heap)
        out_view[count] = index
        heap_dists_view[index] = 0
        for k in range(row_splits[index], row_splits[index+1]):
            i = indices[k]
            di = -dists[k]

            old_di = heap_dists_view[i]
            if di > old_di:
                heap_dists_view[i] = di
                heapq.heappush(heap, (di, i))
    return out
