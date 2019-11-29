from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
# pylint: disable=no-name-in-module
from ifp._rej import rejection_sample_dists_prealloc
# pylint: enable=no-name-in-module


def rejection_sample_dists_prealloc_base(dists, indices, min_dists, consumed,
                                         out):
    count = 0
    in_size = indices.shape[0]
    out_size = out.size
    for i in range(in_size):
        if count >= out_size:
            break
        if not consumed[i]:
            js = indices[i]
            min_dists[js] = np.minimum(min_dists[js], dists[i])
            consumed[js] = True
            out[count] = i
            count += 1
    else:
        i += 1
    return count, i


def rejection_sample_dists_base(dists, indices, max_size):
    in_size = indices.shape[0]
    consumed = np.zeros((in_size,), dtype=np.bool)
    min_dists = np.full((in_size,), np.inf, dtype=np.float32)
    # min_dists = None
    out = np.empty((max_size,), dtype=np.uint32)
    n, _ = rejection_sample_dists_prealloc_base(dists, indices, min_dists,
                                                consumed, out)
    return dists, out[:n]


def rejection_sample_base(indices, max_size):
    in_size = indices.shape[0]
    consumed = np.zeros((in_size,), dtype=np.bool)
    out = []
    for i in range(in_size):
        if len(out) >= max_size:
            break
        if not consumed[i]:
            consumed[indices[i]] = True
            out.append(i)
    return np.array(out, dtype=np.uint32)


def rejection_sample_dists(dists, indices, max_size):
    in_size = indices.shape[0]
    consumed = np.zeros((in_size,), dtype=np.uint8)
    min_dists = np.full((in_size,), np.inf, dtype=np.float32)

    out = np.empty((max_size,), dtype=np.uint32)
    n, _ = rejection_sample_dists_prealloc(dists, indices, min_dists, consumed,
                                           out)
    return dists, out[:n]
