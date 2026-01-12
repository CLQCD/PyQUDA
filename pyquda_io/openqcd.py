from math import isclose
from os import path
import struct
from typing import List

import numpy

from pyquda_comm import getSublatticeSize, openReadHeader, openWriteHeader, readMPIFile, writeMPIFile
from .io_utils import gaugeEvenOdd, gaugeLexico, gaugePlaquette, gaugeOddShiftForward, gaugeEvenShiftBackward

Nd, Ns, Nc = 4, 4, 3


def readGauge(filename: str, plaquette: bool = True, lexico: bool = True):
    filename = path.expanduser(path.expandvars(filename))
    with openReadHeader(filename) as f:
        if f.fp is not None:
            latt_size = list(struct.unpack("<iiii", f.fp.read(16))[::-1])
            plaquette_ = struct.unpack("<d", f.fp.read(8))[0] / Nc
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c16", f.offset

    gauge_reorder = readMPIFile(filename, dtype, offset, (Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), (1, 2, 3, 0))

    gauge = numpy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), dtype)
    for t in range(Lt):
        for y in range(Ly):
            for z in range(Lz):
                for x in range(Lx // 2):
                    x_ = 2 * x + (1 - (t + z + y) % 2)
                    z_ = z // 2
                    gauge[[3, 0, 1, 2], :, t, z, y, x, :, :] = gauge_reorder[t, x_, y, z_]

    gauge = gaugeOddShiftForward(latt_size, gauge)
    if lexico:
        gauge = gaugeLexico([Lx, Ly, Lz, Lt], gauge)
        if plaquette:
            assert isclose(plaquette_, gaugePlaquette(latt_size, gauge))
    else:
        if plaquette:
            assert isclose(plaquette_, gaugePlaquette(latt_size, gaugeLexico([Lx, Ly, Lz, Lt], gauge)))
    gauge = gauge.astype("<c16")

    return latt_size, gauge


def writeGauge(filename: str, latt_size: List[int], gauge: numpy.ndarray, lexico: bool = True):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c16", None

    gauge = gauge.astype(dtype)
    if lexico:
        plaquette = gaugePlaquette(latt_size, gauge)
        gauge = gaugeEvenOdd([Lx, Ly, Lz, Lt], gauge)
    else:
        plaquette = gaugePlaquette(latt_size, gaugeLexico([Lx, Ly, Lz, Lt], gauge))
    gauge = gaugeEvenShiftBackward(latt_size, gauge)
    gauge_reorder = numpy.zeros((Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), dtype)
    for t in range(Lt):
        for y in range(Ly):
            for z in range(Lz):
                for x in range(Lx // 2):
                    x_ = 2 * x + (1 - (t + z + y) % 2)
                    z_ = z // 2
                    gauge_reorder[t, x_, y, z_] = gauge[[3, 0, 1, 2], :, t, z, y, x, :, :]

    gauge = gauge_reorder.astype(dtype)
    with openWriteHeader(filename) as f:
        if f.fp is not None:
            f.fp.write(struct.pack("<iiii", *latt_size[::-1]))
            f.fp.write(struct.pack("<d", plaquette * Nc))
    offset = f.offset

    writeMPIFile(filename, dtype, offset, (Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), (1, 2, 3, 0), gauge)
