from typing import List

from pyquda_comm import _factorization


def real_geo_block_size(sublatt_size: List[int], geo_block_size: List[List[int]]):
    from copy import copy

    X = sublatt_size
    geo_block_size_ = []
    for level in range(len(geo_block_size)):
        geo_bs = copy(geo_block_size[level])
        for d in range(4):
            while geo_bs[d] > 0:
                if d == 0 and X[0] == geo_bs[0]:
                    print(f"X-dimension length {X[0]} cannot block length {geo_bs[0]}")
                elif (X[d] // geo_bs[d] + 1) % 2 == 0:
                    print(f"Indexing does not (yet) support odd coarse dimensions: X({d}) = {X[d] // geo_bs[d]}")
                elif (X[d] // geo_bs[d]) * geo_bs[d] != X[d]:
                    print(f"cannot block dim[{d}]={X[d]} with block size = {geo_bs[d]}")
                else:
                    break
                geo_bs[d] //= 2
            if geo_bs[d] == 0:
                raise ValueError(f"Unable to block dimension {d}")
        X = [x // b for x, b in zip(X, geo_bs)]
        geo_block_size_.append(geo_bs)
    return geo_block_size_


def _partition(factor: List[List[List[int]]], n_level: int, block_size: List[int] = None, idx: int = 0):
    if idx == 0:
        block_size = [1 for _ in range(n_level)]
        factor = _factorization(factor, n_level)
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
