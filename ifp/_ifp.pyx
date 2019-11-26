"""Followed guide at https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial ."""

cimport numpy as np
import numpy as np
cimport cython
import _heapq as heapq
# from _heapq import heappop
# from _heapq import headpush


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def ifp_sample_heap_unchecked(
        float[:, ::1] dists, unsigned int[:, ::1] indices, Py_ssize_t out_size):
    """
    Args:
        dists: [in_size, K] floats
        indices: [in_size, K] ints in [0, in_size)
        out_size: int, (maximum) number of outputs

    Returns:
        indices: [out_size] int indices of sampled points.
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

    cdef float[:] heap_dists_view = heap_dists
    for i in range(in_size):
        heap_dists_view[i] = neg_inf
    cdef int[:] out_view = out

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
