from copy import deepcopy
from typing import List, Optional, Union

from pyquda_comm import _factorization


def real_geo_block_size(sublatt_size: List[int], geo_block_size: List[List[int]]):
    ndim = len(sublatt_size)
    sublatt_size_ = deepcopy(sublatt_size)
    geo_block_size_ = deepcopy(geo_block_size)

    for level in range(len(geo_block_size)):
        X = sublatt_size_
        geo_bs = geo_block_size_[level]

        # Only loop over four dimensions for now, we don't have
        # to worry about the fifth dimension until we hit chiral fermions.
        for d in range(4):
            while geo_bs[d] > 0:
                if d == 0 and X[0] == geo_bs[0]:
                    print("MG level (%d): X-dimension length %d cannot block length %d" % (level, X[0], geo_bs[0]))
                elif (X[d] // geo_bs[d] + 1) % 2 == 0:
                    print(
                        "MG level (%d): Indexing does not (yet) support odd coarse dimensions: X(%d) = %d"
                        % (level, d, X[d] // geo_bs[d])
                    )
                elif (X[d] // geo_bs[d]) * geo_bs[d] != X[d]:
                    print("MG level (%d): cannot block dim[%d]=%d with block size = %d" % (level, d, X[d], geo_bs[d]))
                else:
                    break  # this is a valid block size so let's use it
                geo_bs[d] //= 2
            if geo_bs[d] == 0:
                raise ValueError("MG level (%d): Unable to block dimension %d" % (level, d))

        if ndim > 4:
            raise ValueError("MG level (%d): Number of dimensions %d not supported" % (level, ndim))

        aggregate_size = 1
        for d in range(ndim):
            aggregate_size *= geo_bs[d]

        if aggregate_size == 1:
            raise ValueError("MG level (%d): Invalid MG aggregate size %d, cannot be 1" % (level, aggregate_size))
        if aggregate_size % 2 != 0:
            raise ValueError("MG level (%d): Invalid MG aggregate size %d, must be even" % (level, aggregate_size))
        if aggregate_size > 1024:
            raise ValueError("MG level (%d): Invalid MG aggregate size %d, must be <= 1024" % (level, aggregate_size))

        sublatt_size_ = [x // b for x, b in zip(X, geo_bs)]
    return geo_block_size_


def _partition(
    factor: Union[int, List[List[List[int]]]], n_level: int, block_size: Optional[List[int]] = None, idx: int = 0
):
    if idx == 0:
        assert isinstance(factor, int) and block_size is None
        block_size = [1 for _ in range(n_level)]
        factor = _factorization(factor, n_level)
    assert isinstance(factor, list) and block_size is not None
    if idx == len(factor):
        yield block_size
    else:
        for factor_size in factor[idx]:
            yield from _partition(
                factor,
                n_level,
                [G * f for G, f in zip(block_size, factor_size)],
                idx + 1,
            )
