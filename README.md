# Approximate Iterative Farthest Point Sampling

Cython implementation based on pre-computed distances (e.g. from a `KDTree`) using a priority queue.

## Installation

```bash
pip install Cython
pip install git+git://github.com/jackd/ifp-sample.git
```

Or if you want to make changes:

```bash
git clone https://github.com/jackd/ifp-sample.git
pip install -e ifp-sample
```

## Usage

```python
import numpy as np
from scipy.spatial import cKDTree
from ifp import ifp_sample

coords = np.random.uniform(size=(1024, 3))
dists, indices = cKDTree(coords).query(coords, 8)
sampled_indices = ifp_sample(dists, indices, num_out=512)
sampled_coords = coords[sampled_indices]
```

See [example.py](example.py) for basic benchmark (~3x faster than pure python version).
