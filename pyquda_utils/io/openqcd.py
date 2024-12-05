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

    gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0))
    gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
    return latt_size, plaquette, gauge


def writeGauge(filename: str, latt_size: List[int], plaquette: float, gauge: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = latt_size
    dtype, offset = "<c16", None

    gauge = numpy.ascontiguousarray(gauge.transpose(1, 2, 3, 4, 0, 5, 6).astype(dtype))
    if getMPIRank() == 0:
        with open(filename, "wb") as f:
            f.write(struct.pack("<iiii", *latt_size[::-1]))
            f.write(struct.pack("<d", plaquette * Nc))
            offset = f.tell()
    offset = getMPIComm().bcast(offset)

    writeMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0), gauge)
