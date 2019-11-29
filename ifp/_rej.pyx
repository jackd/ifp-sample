cimport cython
cimport numpy as np
import numpy as np
import heapq


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def rejection_sample_dists_unchecked(
        float[:, :] dists, unsigned int[:, :] indices,
        float[:] min_dists, np.uint8_t[:] consumed,
        unsigned int[:] out):
    """
    Args:
        indices: [in_size, K] uint32 in [0, in_size).
        consumed: [in_size] bool array of consumed inputs.
        out: [max_size] uint32 array to save output.

    Returns:
        num_sampled: int, number of points sampled.
        stop_index: int, number of input points processed.
    """
    cdef Py_ssize_t in_size = indices.shape[0]
    cdef Py_ssize_t K = indices.shape[1]
    cdef Py_ssize_t out_size = indices.shape[0]

    cdef float inf = np.inf
    cdef float dist
    cdef int count = 0
    cdef int i
    cdef int index
    cdef int k
    cdef unsigned int[:] indices_row
    cdef float[:] dists_row

    if out_size == 0:
        return 0, 0

    for i in range(in_size):
        if count >= out_size:
            break
        if not consumed[i]:
            dists_row = dists[i]
            min_dists[i] = 0
            indices_row = indices[i]
            for k in range(1, K):
                index = indices_row[k]
                dist = dists_row[index]
                # if the below condition is false, consumed should already
                # be 1.
                if dist < min_dists[index]:
                    # only mark future points as consumed
                    # means consumed ends up being an inverse mask
                    # if index > i:
                    consumed[index] = 1
                    min_dists[index] = dist
            out[count] = i
            count += 1
    else:
        i += 1

    return count, i


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def rejection_sample_unchecked(
        unsigned int[:, :] indices, np.uint8_t[:] consumed,
        unsigned int[:] out):
    """
    Args:
        indices: [in_size, K] uint32 in [0, in_size).
        consumed: [in_size] bool array of consumed inputs.
        out: [max_size] uint32 array to save output.

    Returns:
        num_sampled: int, number of points sampled.
        stop_index: int, number of input points processed.
    """
    cdef Py_ssize_t in_size = indices.shape[0]
    cdef Py_ssize_t K = indices.shape[1]
    cdef Py_ssize_t out_size = indices.shape[0]

    cdef int count = 0
    cdef int i
    cdef int index
    cdef int k
    cdef unsigned int[:] indices_row

    if out_size == 0:
        return 0, 0

    for i in range(in_size):
        if count >= out_size:
            break
        if not consumed[i]:
            indices_row = indices[i]
            for k in range(1, K):
                index = indices_row[k]
                # only mark future points as consumed
                # means consumed ends up being an inverse mask
                # if index > i:
                consumed[index] = 1
            out[count] = i
            count += 1
    else:
        i += 1

    return count, i


# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
# def rejection_sample_unchecked(
#         unsigned int[:, :] indices, np.uint8_t[:] consumed,
#         unsigned int[:] out):
#     """
#     Args:
#         indices: [in_size, K] uint32 in [0, in_size).
#         consumed: [in_size] bool array of consumed inputs.
#         out: [max_size] uint32 array to save output.

#     Returns:
#         num_sampled: int, number of points sampled.
#         stop_index: int, number of input points processed.
#     """
#     cdef Py_ssize_t in_size = indices.shape[0]
#     cdef Py_ssize_t K = indices.shape[1]
#     cdef Py_ssize_t out_size = indices.shape[0]

#     cdef int count = 0
#     cdef int i

#     if out_size == 0:
#         return 0, 0

#     for i in range(in_size):
#         if count >= out_size:
#             break
#         if not consumed[i]:
#             for j in indices[i]:
#                 consumed[j] = True
#             out[count] = i
#             count += 1
#     else:
#         i += 1

#     return count, i


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def rejection_sample_ragged_unchecked(
        unsigned int[:] indices, unsigned int[:] row_splits,
        np.uint8_t[:] consumed, unsigned int[:] out):
    """
    Args:
        indices: [E] uint32 in [0, in_size).
        row_splits: [in_size + 1] uint32 in [0, in_size).
        consumed: [in_size] bool array of consumed inputs.
        out: [max_size] uint32 array to save output.

    Returns:
        num_sampled: int number of points sampled. Values will be assigned to
            out[:num_sampled]
        stop_index: int number of input points processed.
    """
    cdef Py_ssize_t in_size = row_splits.shape[0] - 1
    cdef Py_ssize_t out_size = indices.shape[0]

    cdef int count = 0
    cdef int i
    cdef int j

    if out_size == 0:
        return 0, 0

    for i in range(in_size):
        if count >= out_size:
                break
        if not consumed[i]:
            for j in range(row_splits[i], row_splits[i+1]):
                consumed[indices[j]] = True
            out[count] = i
            count += 1
    else:
        i += 1
    return count, i



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def rejection_sample_dists_prealloc(
        float[:, :] dists, unsigned int[:, :] indices,
        float[:] min_dists, np.uint8_t[:] consumed, unsigned int[:] out):
    """
    Rejection sampling which also monitors minimum distances.

    Args:
        dists: dists to nearest neighbors
        indices: indices of nearest neighbors
        min_dists: buffer for minimum distance
        consumed: buffer for rejected/selected points
    """

    cdef Py_ssize_t in_size = indices.shape[0]
    cdef Py_ssize_t K = indices.shape[1]
    cdef Py_ssize_t out_size = out.size
    cdef float[:] dists_row
    cdef unsigned int[:] indices_row
    cdef float inf = np.inf
    cdef int count = 0
    cdef int i
    cdef int k
    cdef int index
    cdef float dist
    for i in range(in_size):
        if count >= out_size:
            break
        if consumed[i] == 0:
            indices_row = indices[i]
            dists_row = dists[i]
            min_dists[i] = 0
            for k in range(1, K):
                index = indices_row[k]
                dist = dists_row[k]
                if dist < inf:
                    if index > i:
                        consumed[index] = 1

                    if dist < min_dists[index]:
                        min_dists[index] = dist
            out[count] = i
            count += 1
    else:
        i += 1
    return count, i
