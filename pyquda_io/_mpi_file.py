from typing import Sequence

import numpy
from mpi4py import MPI
from mpi4py.util import dtlib

from pyquda_comm import getMPIComm, getMPISize, getMPIRank, getCoordFromRank, getRankFromCoord


def getSublatticeSize(latt_size: Sequence[int], grid_size: Sequence[int]):
    GLx, GLy, GLz, GLt = latt_size
    Gx, Gy, Gz, Gt = grid_size
    assert GLx % Gx == 0 and GLy % Gy == 0 and GLz % Gz == 0 and GLt % Gt == 0
    return [GLx // Gx, GLy // Gy, GLz // Gz, GLt // Gt]


def getGridCoord(grid_size: Sequence[int]):
    return getCoordFromRank(getMPIRank(), grid_size)


def getNeighbourRank(grid_size: Sequence[int]):
    Gx, Gy, Gz, Gt = grid_size
    gx, gy, gz, gt = getCoordFromRank(getMPIRank(), grid_size)
    return [
        getRankFromCoord([(gx + 1) % Gx, gy, gz, gt], grid_size),
        getRankFromCoord([gx, (gy + 1) % Gy, gz, gt], grid_size),
        getRankFromCoord([gx, gy, (gz + 1) % Gz, gt], grid_size),
        getRankFromCoord([gx, gy, gz, (gt + 1) % Gt], grid_size),
        getRankFromCoord([(gx - 1) % Gx, gy, gz, gt], grid_size),
        getRankFromCoord([gx, (gy - 1) % Gy, gz, gt], grid_size),
        getRankFromCoord([gx, gy, (gz - 1) % Gz, gt], grid_size),
        getRankFromCoord([gx, gy, gz, (gt - 1) % Gt], grid_size),
    ]


def getSubarray(shape: Sequence[int], axes: Sequence[int], grid: Sequence[int]):
    sizes = [d for d in shape]
    subsizes = [d for d in shape]
    starts = [d if i in axes else 0 for i, d in enumerate(shape)]
    coord = getGridCoord(grid)
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
    grid: Sequence[int],
):
    assert getMPISize() == int(numpy.prod(grid))
    sizes, subsizes, starts = getSubarray(shape, axes, grid)
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
    grid: Sequence[int],
    buf: numpy.ndarray,
):
    assert getMPISize() == int(numpy.prod(grid))
    sizes, subsizes, starts = getSubarray(shape, axes, grid)
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")
    buf = buf.view(native_dtype)

    fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    filetype = dtlib.from_numpy_dtype(native_dtype).Create_subarray(sizes, subsizes, starts)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Write_all(buf)
    filetype.Free()
    fh.Close()
