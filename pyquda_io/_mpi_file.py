from typing import Sequence

import numpy
from mpi4py import MPI
from mpi4py.util import dtlib


def getSublatticeSize(latt_size: Sequence[int], grid_size: Sequence[int]):
    GLx, GLy, GLz, GLt = latt_size
    Gx, Gy, Gz, Gt = grid_size
    assert GLx % Gx == 0 and GLy % Gy == 0 and GLz % Gz == 0 and GLt % Gt == 0
    return [GLx // Gx, GLy // Gy, GLz // Gz, GLt // Gt]


def getGridCoord(grid_size: Sequence[int]):
    rank = MPI.COMM_WORLD.Get_rank()
    Gx, Gy, Gz, Gt = grid_size
    gx, gy, gz, gt = rank // Gt // Gz // Gy % Gx, rank // Gt // Gz % Gy, rank // Gt % Gz, rank % Gt
    return [gx, gy, gz, gt]


def getNeighbourRank(grid_size: Sequence[int]):
    Gx, Gy, Gz, Gt = grid_size
    gx, gy, gz, gt = getGridCoord(grid_size)
    return [
        (((gx + 1) % Gx * Gy + gy) * Gz + gz) * Gt + gt,
        ((gx * Gy + (gy + 1) % Gy) * Gz + gz) * Gt + gt,
        ((gx * Gy + gy) * Gz + (gz + 1) % Gz) * Gt + gt,
        ((gx * Gy + gy) * Gz + gz) * Gt + (gt + 1) % Gt,
        (((gx - 1) % Gx * Gy + gy) * Gz + gz) * Gt + gt,
        ((gx * Gy + (gy - 1) % Gy) * Gz + gz) * Gt + gt,
        ((gx * Gy + gy) * Gz + (gz - 1) % Gz) * Gt + gt,
        ((gx * Gy + gy) * Gz + gz) * Gt + (gt - 1) % Gt,
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
    assert MPI.COMM_WORLD.Get_size() == int(numpy.prod(grid))
    sizes, subsizes, starts = getSubarray(shape, axes, grid)
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")
    buf = numpy.empty(subsizes, native_dtype)

    fh = MPI.File.Open(MPI.COMM_WORLD, filename, MPI.MODE_RDONLY)
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
    assert MPI.COMM_WORLD.Get_size() == int(numpy.prod(grid))
    sizes, subsizes, starts = getSubarray(shape, axes, grid)
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")
    buf = buf.view(native_dtype)

    fh = MPI.File.Open(MPI.COMM_WORLD, filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    filetype = dtlib.from_numpy_dtype(native_dtype).Create_subarray(sizes, subsizes, starts)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Write_all(buf)
    filetype.Free()
    fh.Close()
