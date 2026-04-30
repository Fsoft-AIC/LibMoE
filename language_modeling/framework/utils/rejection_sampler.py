import numpy as np
from typing import Callable, List, Any, Tuple, Optional, Dict
import multiprocessing
import math
from tqdm import tqdm
from .parallel_map import ParallelMapPool


def rejection_sample(n_samples: int, seed: np.random.RandomState, sampler: Callable[[np.random.RandomState], Any],
                     test: Callable[[Any], bool] = lambda x: True, get_hash: Callable[[Any], int] = hash, exclude=set(),
                     pbar: Optional[tqdm] = None) -> List[Any]:

    def get_samples(seed):
        seed = np.random.RandomState(seed)
        res = []
        for _ in range(100):
            res.append(sampler(seed))
        return res

    data = []

    nproc = multiprocessing.cpu_count()

    seed_start = seed.randint(0x000FFFFF)
    seed_offset = 0

    known = exclude.copy()

    pbar = pbar if pbar is not None else tqdm(total = n_samples)
    with ParallelMapPool(get_samples) as ppool:
        while len(data) < n_samples:
            seeds = [seed_start + seed_offset * nproc + i for i in range(nproc)]
            seed_offset += nproc

            res = ppool.map(seeds)
            res = sum(res, [])

            for s in res:
                h = get_hash(s)

                # Ensure there are no repeats
                if h in known:
                    continue

                # Rejection
                if not test(s):
                    continue

                known.add(h)

                data.append(s)
                pbar.update(1)

                if len(data) >= n_samples:
                    break

    return data


def rejection_sample_length_buckets(n_samples: int, length_range: Tuple[int, int],
        seed: np.random.RandomState, sampler: Callable[[int], Any],
        get_length: Callable[[Any], int], get_hash: Callable[[Any], int] = hash, exclude=set(),
        limits: Optional[Dict[int, int]] = None, test: Callable[[Any], bool] = lambda x: True) -> List[Any]:

    if limits is None:
        limits = {}

    bins = {}
    remaining = n_samples
    for i in range(length_range[0], length_range[1] + 1):
        lim = math.ceil(remaining / (length_range[1] - i + 1))
        a = min(lim, int(limits.get(i, lim)))
        bins[i] = a
        remaining -= a

    assert sum(bins.values()) >= n_samples

    def test_length(sample) -> bool:
        len = get_length(sample)
        if bins.get(len, 0) > 0 and test(sample):
            bins[len] -= 1
            return True
        else:
            return False

    return rejection_sample(n_samples, seed, sampler, test_length, get_hash, exclude)
