from os import path
from typing import List, Tuple

import numpy
from numpy.lib.format import read_magic, read_array_header_1_0, write_array_header_1_0

from .mpi_utils import getSublatticeSize, openReadHeader, openWriteHeader, readMPIFile, writeMPIFile

Nd, Ns, Nc = 4, 4, 3


def read_array_header(filename: str) -> Tuple[Tuple[int, ...], str, int]:
    with openReadHeader(filename) as f:
        if f.fp is not None:
            assert read_magic(f.fp) == (1, 0)
            shape, fortran_order, dtype = read_array_header_1_0(f.fp)
            assert not fortran_order
    return shape, dtype.str, f.offset


def write_array_header(filename: str, shape: Tuple[int, ...], dtype: str) -> int:
    with openWriteHeader(filename) as f:
        if f.fp is not None:
            d = {"shape": shape, "fortran_order": False, "descr": dtype}
            write_array_header_1_0(f.fp, d)
    return f.offset


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
