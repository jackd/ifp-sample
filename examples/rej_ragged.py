from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial import cKDTree as KDTree  # pylint: disable=no-name-in-module
from timeit import timeit
import functools

from ifp import rejection_sample_ragged

k = 16
in_size = 1024
max_size = 1024
max_dist = 0.05


def rejection_sample_ragged_base(indices: np.ndarray, row_splits: np.ndarray,
                                 max_size: int) -> np.ndarray:
    N = row_splits.size - 1
    consumed = np.zeros((N,), dtype=np.bool)
    out = []
    for i in range(N):
        if len(out) >= max_size:
            break
        if not consumed[i]:
            consumed[indices[row_splits[i]:row_splits[i + 1]]] = True
            out.append(i)
    return np.array(out, dtype=np.uint32)


np.random.seed(123)
x = np.random.uniform(size=(in_size, 2)).astype(dtype=np.float32)
tree = KDTree(x)
dists, indices = tree.query(x, k)
valid = dists < max_dist
indices = indices[valid]
row_lengths = np.count_nonzero(valid, axis=1)
row_splits = np.pad(np.cumsum(row_lengths), [[1, 0]], 'constant')

kwargs = dict(indices=indices, row_splits=row_splits, max_size=max_size)

num_runs = 100
print('cython implementation')
print(
    timeit(functools.partial(rejection_sample_ragged, **kwargs),
           number=num_runs) / num_runs)
print('python implementation')
print(
    timeit(functools.partial(rejection_sample_ragged_base, **kwargs),
           number=num_runs) / num_runs)

sample_indices = rejection_sample_ragged(**kwargs)
base_indices = rejection_sample_ragged_base(**kwargs)
print('Difference: {}'.format(np.any(sample_indices != base_indices)))
print('out_size = {}'.format(len(base_indices)))

import matplotlib.pyplot as plt

x_out = x[sample_indices]
plt.scatter(*x.T, c='k')
plt.scatter(*(x_out.T), c='r')
plt.show()
