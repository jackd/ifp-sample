"""Followed guide at https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial ."""

cimport numpy as np
import numpy as np
import heapq
cimport cython


# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
def ifp_sample_heap_unchecked(
        float[:, ::1] dists, unsigned int[:, ::1] indices, int out_size):
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

    out = -np.ones((out_size,), dtype=np.intc)
    heap_dists = np.empty((in_size,), dtype=np.float32)
    heap_dists[:] = -np.inf
    shuffled_indices = np.arange(in_size, dtype=np.intc)
    np.random.shuffle(shuffled_indices)

    cdef float[:] heap_dists_view = heap_dists
    cdef int[:] out_view = out
    cdef int[:] shuffled_indices_view = shuffled_indices

    heap = list(zip(heap_dists_view, shuffled_indices_view))

    while heap:
        dist, index = heapq.heappop(heap)
        if dist != heap_dists_view[index]:
            continue
        out_view[count] = index
        count += 1
        if count >= out_size:
            break

        for k in range(1, K):
            i = indices[index, k]
            di = -dists[index, k]

            old_di = heap_dists_view[i]
            heap_dists_view[index] = 0
            if di > old_di:
                heap_dists_view[i] = di
                heapq.heappush(heap, (di, i))
    else:
        raise RuntimeError('Should have broken...')
    return out
