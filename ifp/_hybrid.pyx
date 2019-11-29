cimport cython
cimport numpy as np
import numpy as np
import heapq
from ifp._ifp import ifp_sample_heap_progressive_unchecked
from ifp._rej import rejection_sample_dists_prealloc


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def ifp_sample_hybrid_prealloc(
        float[:, :] dists, unsigned int[:, :] indices, Py_ssize_t out_size,
        float[:] min_dists, np.uint8_t[:] consumed, unsigned int[:] out):
    cdef Py_ssize_t in_size = indices.shape[0]
    cdef int n
    n, _ = rejection_sample_dists_prealloc(dists, indices, min_dists, consumed,
                                            out)
    if n == out_size:
        return
    cdef list heap = [(-d, i) for i, d in enumerate(min_dists) if d > 0]
    heapq.heapify(heap)

    ifp_sample_heap_progressive_unchecked(
        dists, indices, min_dists, out[n:], heap)
