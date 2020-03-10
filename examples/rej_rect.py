from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial import cKDTree as KDTree  # pylint: disable=no-name-in-module
from timeit import timeit
import functools
# import numba

from ifp import rejection_sample
from ifp.rej import rejection_sample_dists
from ifp.rej import rejection_sample_dists_base
from ifp.rej import rejection_sample_base
from numba import jit

from benchmark_util import run_benchmarks


@jit(nopython=True)
def _rejection_sample_numba(indices, consumed, out):
    in_size = indices.shape[0]
    max_size = out.shape[0]
    count = 0
    for i in range(in_size):
        if count >= max_size:
            break
        if not consumed[i]:
            consumed[indices[i]] = True
            out[count] = i
            count += 1
    return out[:count]


def rejection_sample_numba(indices, max_size):
    in_size = indices.shape[0]
    consumed = np.zeros((in_size,), dtype=np.bool)
    out = np.empty((max_size,), dtype=np.uint32)
    return _rejection_sample_numba(indices, consumed, out)


k = 16
in_size = 1024
max_size = 512
num_iters = 200
burn_iters = 50
np.random.seed(123)

x = np.random.uniform(size=(in_size, 2)).astype(dtype=np.float32)
tree = KDTree(x)
dists, indices = tree.query(x, k)
dists = dists.astype(np.float32)
indices = indices.astype(np.uint32)

kwargs = dict(indices=indices, max_size=max_size)

rej = functools.partial(rejection_sample, **kwargs)
rej_dist = functools.partial(rejection_sample_dists, dists=dists, **kwargs)
rej_base = functools.partial(rejection_sample_base, **kwargs)
rej_dist_base = functools.partial(rejection_sample_dists_base,
                                  dists=dists,
                                  **kwargs)
rej_numba = functools.partial(rejection_sample_numba, **kwargs)
run_benchmarks(
    burn_iters,
    num_iters,
    ('rej', rej),
    ('rej_dist', rej_dist),
    ('rej_base', rej_base),
    ('rej_dist_base', rej_dist_base),
    ('rej_numba', rej_numba),
)

rej_indices = rej()
base_indices = rej_base()
numba_indices = rej_numba()
_, alt_indices = rej_dist()
print('All same (rej vs rej base) : {}'.format(
    np.all(rej_indices == base_indices)))
print('All same (rej vs rej numba) : {}'.format(
    np.all(rej_indices == numba_indices)))
print('All same (rej-dist vs base): {}'.format(
    np.any(alt_indices == base_indices)))
print(alt_indices.shape[0])
import matplotlib.pyplot as plt

x_out = x[base_indices]
plt.scatter(*x.T, c='k')
plt.scatter(*(x_out.T), c='r')
plt.show()
