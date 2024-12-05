from os import path
import struct
from typing import List

import numpy

from pyquda import getSublatticeSize, getMPIRank, getMPIComm, readMPIFile, writeMPIFile

Nd, Ns, Nc = 4, 4, 3


def readGauge(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        latt_size = struct.unpack("<iiii", f.read(16))[::-1]
        plaquette = struct.unpack("<d", f.read(8))[0] / Nc
        offset = f.tell()
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype = "<c16"

    gauge_reorder = readMPIFile(filename, dtype, offset, (Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), (1, 2, 3, 0))

    gauge = numpy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), dtype="<c16")
    for t in range(Lt):
        for y in range(Ly):
            for z in range(Lz):
                for x in range(Lx // 2):
                    x_ = 2 * x + (1 - (t + z + y) % 2)
                    z_ = z // 2
                    gauge[[3, 0, 1, 2], :, t, z, y, x, :, :] = gauge_reorder[t, x_, y, z_]

    gauge = gauge.astype("<c16")
    return latt_size, plaquette, gauge


def writeGauge(filename: str, latt_size: List[int], plaquette: float, gauge: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c16", None

    gauge_reorder = numpy.zeros((Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), dtype="<c16")
    for t in range(Lt):
        for y in range(Ly):
            for z in range(Lz):
                for x in range(Lx // 2):
                    x_ = 2 * x + (1 - (t + z + y) % 2)
                    z_ = z // 2
                    gauge_reorder[t, x_, y, z_] = gauge[[3, 0, 1, 2], :, t, z, y, x, :, :]

    gauge = gauge_reorder.astype(dtype)
    if getMPIRank() == 0:
        with open(filename, "wb") as f:
            f.write(struct.pack("<iiii", *latt_size[::-1]))
            f.write(struct.pack("<d", plaquette * Nc))
            offset = f.tell()
    offset = getMPIComm().bcast(offset)

    writeMPIFile(filename, dtype, offset, (Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), (1, 2, 3, 0), gauge)
