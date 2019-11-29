from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial import cKDTree as KDTree  # pylint: disable=no-name-in-module
from timeit import timeit
import functools

from ifp import ifp_sample
from ifp import ifp_sample_ragged

k = 32
in_size = 1024
out_size = 512
max_dist = 0.05
num_runs = 100

np.random.seed(123)
x = np.random.uniform(size=(in_size, 2)).astype(dtype=np.float32)
tree = KDTree(x)
dists, indices = tree.query(x, k)

dists[dists > max_dist] = np.inf

fn = functools.partial(ifp_sample,
                       indices=indices,
                       dists=dists,
                       out_size=out_size)
print('padded')
print(timeit(fn, number=num_runs) / num_runs)

out = fn()

mask = dists < max_dist
row_lengths = np.count_nonzero(mask, axis=1)
row_splits = np.cumsum(np.concatenate([[0], row_lengths], axis=0))

dists = dists[mask]
indices = indices[mask]

fn = functools.partial(ifp_sample_ragged,
                       indices=indices,
                       dists=dists,
                       row_splits=row_splits,
                       out_size=out_size)
print(row_splits[-1] / (in_size * 32))
print('ragged')
print(timeit(fn, number=num_runs) / num_runs)

out_ragged = fn()
print(len(set(out).symmetric_difference(set(out_ragged))))

x_out = x[out_ragged]

import matplotlib.pyplot as plt
plt.scatter(*x.T, c='k')
plt.scatter(*(x_out.T), c='r')
plt.show()
