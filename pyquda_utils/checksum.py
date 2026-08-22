from typing import Tuple
from zlib import crc32

import numpy
from numpy.typing import NDArray
from mpi4py import MPI

from pyquda_comm import getMPIComm
from pyquda_comm.field import LatticeInfo


def checksumSciDAC(field) -> Tuple[int, int]:
    buf: NDArray = field.lexico().reshape(field.latt_info.volume, -1).view("<u4")
    latt_info: LatticeInfo = field.latt_info

    work = numpy.empty((latt_info.volume), "<u4")
    for i in range(latt_info.volume):
        work[i] = crc32(buf[i])
    sublatt_slice = tuple(slice(g * L, (g + 1) * L) for g, L in zip(latt_info.grid_coord[::-1], latt_info.size[::-1]))
    rank = (
        numpy.arange(latt_info.global_volume, dtype="<u8")
        .reshape(*latt_info.global_size[::-1])[sublatt_slice]
        .reshape(-1)
    )
    rank29 = (rank % 29).astype("<u4")
    rank31 = (rank % 31).astype("<u4")
    sum29 = getMPIComm().allreduce(numpy.bitwise_xor.reduce(work << rank29 | work >> (32 - rank29)).item(), MPI.BXOR)
    sum31 = getMPIComm().allreduce(numpy.bitwise_xor.reduce(work << rank31 | work >> (32 - rank31)).item(), MPI.BXOR)
    return sum29, sum31
