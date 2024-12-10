from os import path
import struct
from typing import List

import numpy
from mpi4py import MPI

from .mpi_file import getSublatticeSize, readMPIFile, writeMPIFile
from .gauge_utils import gaugeEvenOdd, gaugeLexico, gaugeLexicoPlaquette, gaugeOddShiftForward, gaugeEvenShiftBackward

Nd, Ns, Nc = 4, 4, 3


def readGauge(filename: str, grid_size: List[int], plaquette: bool = True, lexico: bool = True):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        latt_size = struct.unpack("<iiii", f.read(16))[::-1]
        plaquette = struct.unpack("<d", f.read(8))[0] / Nc
        offset = f.tell()
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    dtype = "<c16"

    gauge_reorder = readMPIFile(filename, dtype, offset, (Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), (1, 2, 3, 0), grid_size)

    gauge = numpy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), dtype)
    for t in range(Lt):
        for y in range(Ly):
            for z in range(Lz):
                for x in range(Lx // 2):
                    x_ = 2 * x + (1 - (t + z + y) % 2)
                    z_ = z // 2
                    gauge[[3, 0, 1, 2], :, t, z, y, x, :, :] = gauge_reorder[t, x_, y, z_]

    gauge = gaugeOddShiftForward(latt_size, grid_size, gauge)
    if lexico:
        gauge = gaugeLexico([Lx, Ly, Lz, Lt], gauge)
        if plaquette:
            assert numpy.isclose(gaugeLexicoPlaquette(latt_size, grid_size, gauge)[0], plaquette)
    elif plaquette:
        gauge_lexico = gaugeLexico([Lx, Ly, Lz, Lt], gauge)
        assert numpy.isclose(gaugeLexicoPlaquette(latt_size, grid_size, gauge_lexico)[0], plaquette)
    gauge = gauge.astype("<c16")

    return latt_size, gauge


def writeGauge(
    filename: str,
    latt_size: List[int],
    grid_size: List[int],
    gauge: numpy.ndarray,
    plaquette: float = None,
    lexico: bool = True,
):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    dtype, offset = "<c16", None

    gauge = gauge.astype(dtype)
    if lexico:
        if plaquette is None:
            plaquette = gaugeLexicoPlaquette(latt_size, grid_size, gauge)[0]
        gauge = gaugeEvenOdd([Lx, Ly, Lz, Lt], gauge)
    elif plaquette is None:
        gauge_lexico = gaugeLexico([Lx, Ly, Lz, Lt], gauge)
        plaquette = gaugeLexicoPlaquette(latt_size, grid_size, gauge_lexico)[0]
    gauge = gaugeEvenShiftBackward(latt_size, grid_size, gauge)
    gauge_reorder = numpy.zeros((Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), dtype)
    for t in range(Lt):
        for y in range(Ly):
            for z in range(Lz):
                for x in range(Lx // 2):
                    x_ = 2 * x + (1 - (t + z + y) % 2)
                    z_ = z // 2
                    gauge_reorder[t, x_, y, z_] = gauge[[3, 0, 1, 2], :, t, z, y, x, :, :]

    gauge = gauge_reorder.astype(dtype)
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(filename, "wb") as f:
            f.write(struct.pack("<iiii", *latt_size[::-1]))
            f.write(struct.pack("<d", plaquette * Nc))
            offset = f.tell()
    offset = MPI.COMM_WORLD.bcast(offset)

    writeMPIFile(filename, dtype, offset, (Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), (1, 2, 3, 0), grid_size, gauge)
