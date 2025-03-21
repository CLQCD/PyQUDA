from os import path
from typing import List

import numpy
from numpy.lib.format import dtype_to_descr, read_magic, read_array_header_1_0, write_array_header_1_0

from ._mpi_file import getMPIComm, getMPIRank, getSublatticeSize, readMPIFile, writeMPIFile

Nd, Ns, Nc = 4, 4, 3


def _readHeader(filename: str):
    with open(filename, "rb") as f:
        assert read_magic(f) == (1, 0)
        shape, fortran_order, dtype = read_array_header_1_0(f)
        dtype = dtype_to_descr(dtype)
        assert not fortran_order and dtype == "<c16"
        offset = f.tell()
    return shape, dtype, offset


def readGauge(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    shape, dtype, offset = _readHeader(filename)
    latt_size = [shape[i] for i in [4, 3, 2, 1]]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)

    gauge = readMPIFile(filename, dtype, offset, (Nd, Lt, Lz, Ly, Lx, Nc, Nc), (4, 3, 2, 1))
    return latt_size, gauge


def writeGauge(filename: str, latt_size: List[int], gauge: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    if getMPIRank() == 0:
        with open(filename, "wb") as f:
            write_array_header_1_0(
                f, {"shape": (Nd, *latt_size[::-1], Nc, Nc), "fortran_order": False, "descr": "<c16"}
            )
    getMPIComm().Barrier()
    shape, dtype, offset = _readHeader(filename)

    writeMPIFile(filename, dtype, offset, (Nd, Lt, Lz, Ly, Lx, Nc, Nc), (4, 3, 2, 1), gauge)


def readPropagator(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    shape, dtype, offset = _readHeader(filename)
    latt_size = [shape[i] for i in [3, 2, 1, 0]]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)

    propagator = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), (3, 2, 1, 0))
    return latt_size, propagator


def writePropagator(filename: str, latt_size: List[int], propagator: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    if getMPIRank() == 0:
        with open(filename, "wb") as f:
            write_array_header_1_0(
                f, {"shape": (*latt_size[::-1], Ns, Ns, Nc, Nc), "fortran_order": False, "descr": "<c16"}
            )
    getMPIComm().Barrier()
    shape, dtype, offset = _readHeader(filename)

    writeMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), (3, 2, 1, 0), propagator)
