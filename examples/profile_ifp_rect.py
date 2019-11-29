"""
To use, add # cython: profile=True to the top of each pyx file and recompile.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial import cKDTree as KDTree  # pylint: disable=no-name-in-module
import heapq
from timeit import timeit
import functools

from ifp import ifp_sample
from ifp import ifp_sample_hybrid
import pstats, cProfile

k = 8
in_size = 1024
out_size = 256
num_runs = 1000

np.random.seed(123)
x = np.random.uniform(size=(in_size, 2)).astype(dtype=np.float32)
tree = KDTree(x)
dists, indices = tree.query(x, k)

kwargs = dict(indices=indices, dists=dists, out_size=out_size)


def do_it():
    for _ in range(num_runs):
        ifp_sample(**kwargs)
        # ifp_sample_hybrid(**kwargs)


path = '/tmp/profile.prof'
cProfile.runctx('do_it()', globals(), locals(), path)

s = pstats.Stats(path)
s.strip_dirs().sort_stats('cumtime').print_stats()
