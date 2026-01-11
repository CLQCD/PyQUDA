from os import path
from typing import List

import numpy

from pyquda_comm import getSublatticeSize, readMPIFile, writeMPIFile
from pyquda_comm.field import read_array_header, write_array_header

Nd, Ns, Nc = 4, 4, 3


def readGauge(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    shape, dtype, offset = read_array_header(filename)
    assert dtype == "<c16"
    latt_size = [shape[i] for i in [4, 3, 2, 1]]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)

    gauge = readMPIFile(filename, dtype, offset, (Nd, Lt, Lz, Ly, Lx, Nc, Nc), (4, 3, 2, 1))
    return latt_size, gauge


def writeGauge(filename: str, latt_size: List[int], gauge: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c16", None
    offset = write_array_header(filename, (Nd, *latt_size[::-1], Nc, Nc), dtype)

    writeMPIFile(filename, dtype, offset, (Nd, Lt, Lz, Ly, Lx, Nc, Nc), (4, 3, 2, 1), gauge)


def readPropagator(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    shape, dtype, offset = read_array_header(filename)
    assert dtype == "<c16"
    latt_size = [shape[i] for i in [3, 2, 1, 0]]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)

    propagator = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), (3, 2, 1, 0))
    return latt_size, propagator


def writePropagator(filename: str, latt_size: List[int], propagator: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c16", None
    offset = write_array_header(filename, (*latt_size[::-1], Ns, Ns, Nc, Nc), dtype)

    writeMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), (3, 2, 1, 0), propagator)
