from typing import Sequence

import numpy
from mpi4py import MPI
from mpi4py.util import dtlib

from pyquda_comm import (
    initGrid,
    isGridInitialized,
    getMPIComm,
    getMPIRank,
    getGridSize,
    getGridCoord,
    getCoordFromRank,
    getRankFromCoord,
)


def getSublatticeSize(latt_size: Sequence[int], evenodd: bool = True):
    if not isGridInitialized():
        initGrid(None, latt_size, evenodd)
    GLx, GLy, GLz, GLt = latt_size
    Gx, Gy, Gz, Gt = getGridSize()
    if evenodd:
        assert GLx % (2 * Gx) == 0 and GLy % (2 * Gy) == 0 and GLz % (2 * Gz) == 0 and GLt % (2 * Gt) == 0
    else:
        assert GLx % Gx == 0 and GLy % Gy == 0 and GLz % Gz == 0 and GLt % Gt == 0
    return [GLx // Gx, GLy // Gy, GLz // Gz, GLt // Gt]


def getNeighbourRank():
    Gx, Gy, Gz, Gt = getGridSize()
    gx, gy, gz, gt = getCoordFromRank(getMPIRank())
    return [
        getRankFromCoord([(gx + 1) % Gx, gy, gz, gt]),
        getRankFromCoord([gx, (gy + 1) % Gy, gz, gt]),
        getRankFromCoord([gx, gy, (gz + 1) % Gz, gt]),
        getRankFromCoord([gx, gy, gz, (gt + 1) % Gt]),
        getRankFromCoord([(gx - 1) % Gx, gy, gz, gt]),
        getRankFromCoord([gx, (gy - 1) % Gy, gz, gt]),
        getRankFromCoord([gx, gy, (gz - 1) % Gz, gt]),
        getRankFromCoord([gx, gy, gz, (gt - 1) % Gt]),
    ]


def getSubarray(shape: Sequence[int], axes: Sequence[int]):
    sizes = [d for d in shape]
    subsizes = [d for d in shape]
    starts = [d if i in axes else 0 for i, d in enumerate(shape)]
    grid = getGridSize()
    coord = getGridCoord()
    for j, i in enumerate(axes):
        sizes[i] *= grid[j]
        starts[i] *= coord[j]
    return sizes, subsizes, starts


def readMPIFile(
    filename: str,
    dtype: str,
    offset: int,
    shape: Sequence[int],
    axes: Sequence[int],
):
    sizes, subsizes, starts = getSubarray(shape, axes)
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")
    buf = numpy.empty(subsizes, native_dtype)

    fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_RDONLY)
    filetype = dtlib.from_numpy_dtype(native_dtype).Create_subarray(sizes, subsizes, starts)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Read_all(buf)
    filetype.Free()
    fh.Close()

    return buf.view(dtype)


def writeMPIFile(
    filename: str,
    dtype: str,
    offset: int,
    shape: Sequence[int],
    axes: Sequence[int],
    buf: numpy.ndarray,
):
    sizes, subsizes, starts = getSubarray(shape, axes)
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")
    buf = buf.view(native_dtype)

    fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    filetype = dtlib.from_numpy_dtype(native_dtype).Create_subarray(sizes, subsizes, starts)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Write_all(buf)
    filetype.Free()
    fh.Close()
