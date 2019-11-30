"""Followed guide at https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial ."""

cimport cython
import numpy as np
import heapq


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def ifp_sample_heap_progressive_unchecked(
        float[:, :] dists, unsigned int[:, :] indices,
        float[:] min_dists, unsigned int[:] out, list heap):
    """
    Args:
        dists: [in_size, K] float32.
        indices: [in_size, K] uint32 in [0, in_size).
        min_dists: [in_size] progressive minimum distances.
        out: [out_size] array for saving output indices.
        heap: current heap (e.g. returned by heapq.heapify). Note priority
            values on here should be negative distance measures, e.g.
            from heapq.heapify(
                [(-di, i) for i in enumerate(min_dists) if di != 0])
    """
    cdef Py_ssize_t in_size = indices.shape[0]
    cdef Py_ssize_t out_size = out.shape[0]
    cdef Py_ssize_t K = indices.shape[1]
    cdef int count
    cdef unsigned int index
    cdef float dist
    cdef float di
    cdef float old_di
    cdef int k

    cdef unsigned int[:] index_row
    cdef float[:] dists_row

    for count in range(out_size):
        dist, index = heapq.heappop(heap)
        while -dist != min_dists[index]:
            dist, index = heapq.heappop(heap)
        out[count] = index
        min_dists[index] = 0
        index_row = indices[index]
        dists_row = dists[index]
        for k in range(1, K):
            index = index_row[k]
            di = dists_row[k]

            old_di = min_dists[index]
            if di < old_di:
                min_dists[index] = di
                heapq.heappush(heap, (-di, index))


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def ifp_sample_heap_unchecked(
        float[:, :] dists, unsigned int[:, :] indices, Py_ssize_t out_size):
    """
    Args:
        dists: [in_size, K] float32
        indices: [in_size, K] uint32 in [0, in_size)
        out_size: int, (maximum) number of outputs

    Returns:
        indices: [out_size] uint32 indices of sampled points.
    """
    cdef Py_ssize_t in_size = indices.shape[0]

    cdef float neg_inf = -np.inf

    out = np.empty((out_size,), dtype=np.uintc)
    cdef unsigned int[::1] out_view = out
    cdef float[::1] min_dists = np.full(
        (in_size,), np.inf, dtype=np.float32)

    # no need to heapify - all priorities equal
    heap = [(neg_inf, i) for i in range(in_size)]

    ifp_sample_heap_progressive_unchecked(
        dists, indices, min_dists, out_view, heap)
    return out


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def ifp_sample_heap_ragged_unchecked(
        float[:] dists, unsigned int[:] indices,
        unsigned int[:] row_splits,  Py_ssize_t out_size):
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
