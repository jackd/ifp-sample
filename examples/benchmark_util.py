import numpy as np
from timeit import timeit


def run_benchmarks(burn_iters, num_iters, *names_and_fns):
    times = []
    names, fns = zip(*names_and_fns)
    for fn in fns:
        for _ in range(burn_iters):
            fn()
        times.append(timeit(fn, number=num_iters) / num_iters)

    indices = np.argsort(times)
    t0 = times[indices[0]]
    for i in indices:
        print('{:10}: {:.10f} ({:.2f})'.format(names[i], times[i],
                                               times[i] / t0))
