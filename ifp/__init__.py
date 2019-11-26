from ifp._ifp import ifp_sample_heap_unchecked  # pylint: disable=no-name-in-module
import numpy as np


def ifp_sample(dists, indices, out_size):
    """
    Iterative farthest point sampling with pre-calculated distances using heapq.

    Example usage:

    ```python
    import numpy as np
    from scipy.spatial import cKDTree
    from ifp import ifp_sample

    coords = np.random.uniform(size=(1024, 3))
    dists, indices = cKDTree(coords).query(coords, 8)
    sampled_indices = ifp_sample(dists, indices, num_out=512)
    ```

    Args:
        dists: [in_size, K] floats
        indices: [in_size, K] ints in [0, in_size)
        out_size: int, (maximum) number of outputs

    Returns:
        indices: [out_size] uint32 indices of sampled points.
    """
    dists = np.ascontiguousarray(dists)
    indices = np.ascontiguousarray(indices)
    in_size = indices.shape[0]
    if out_size > in_size:
        raise ValueError(
            'out_size cannot be greater than input size, but {} > {}'.format(
                out_size, in_size))
    if len(indices.shape) != 2:
        raise ValueError('indices must be rank 2, got shape {}'.format(
            indices.shape))
    if dists.shape != indices.shape:
        raise ValueError('indices.shape must be the same as dists.shape, '
                         'but {} != {}'.format(indices.shape, dists.shape))
    if np.any(indices >= in_size):
        raise ValueError(
            'indices must all be less than in_size but got max value {} >= {}'.
            format(np.max(indices), in_size))
    return ifp_sample_heap_unchecked(dists.astype(np.float32),
                                     indices.astype(np.uint32),
                                     out_size=out_size)
