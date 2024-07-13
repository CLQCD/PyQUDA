from os import path
from typing import List

import numpy

from ... import getSublatticeSize, readMPIFile, writeMPIFile

Ns, Nc = 4, 3


def readPropagator(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", 0

    propagator = readMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2))
    propagator = propagator.transpose(2, 3, 4, 5, 6, 0, 7, 1).astype("<c16")
    return propagator


def writePropagator(filename: str, propagator: numpy.ndarray, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", 0

    propagator = propagator.astype(dtype).transpose(5, 7, 0, 1, 2, 3, 4, 6).copy()
    writeMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2), propagator)


def readStaggeredPropagator(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", 0

    propagator = readMPIFile(filename, dtype, offset, (Nc, Lt, Lz, Ly, Lx, Nc), (4, 3, 2, 1))
    propagator = propagator.transpose(1, 2, 3, 4, 5, 0).astype("<c16")
    return propagator


def writeStaggeredPropagator(filename: str, propagator: numpy.ndarray, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", 0

    propagator = propagator.astype(dtype).transpose(5, 0, 1, 2, 3, 4).copy()
    writeMPIFile(filename, dtype, offset, (Nc, Lt, Lz, Ly, Lx, Nc), (4, 3, 2, 1), propagator)


def readPropagatorFast(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", 0

    propagator = readMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2))
    return propagator


def writePropagatorFast(filename: str, propagator: numpy.ndarray, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", 0

    writeMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2), propagator)
