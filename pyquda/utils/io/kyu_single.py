from os import path
from typing import List

import numpy

from ... import getSublatticeSize, readMPIFile, writeMPIFile

Ns, Nc = 4, 3


def fromPropagatorFile(filename: str, offset: int, dtype: str, sublatt_size: List[int]):
    Lx, Ly, Lz, Lt = sublatt_size

    propagator_raw = readMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2))
    propagator_raw = propagator_raw.transpose(2, 3, 4, 5, 6, 0, 7, 1).astype("<c16")

    return propagator_raw


def toPropagatorFile(filename: str, offset: int, propagator_raw: numpy.ndarray, dtype: str, sublatt_size: List[int]):
    Lx, Ly, Lz, Lt = sublatt_size

    propagator_raw = propagator_raw.astype(dtype).transpose(5, 7, 0, 1, 2, 3, 4, 6).copy()
    writeMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2), propagator_raw)


def readPropagator(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    sublatt_size = getSublatticeSize(latt_size)
    propagator_raw = fromPropagatorFile(filename, 0, "<c8", sublatt_size)
    return propagator_raw


def writePropagator(filename: str, propagator: numpy.ndarray, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    sublatt_size = getSublatticeSize(latt_size)

    toPropagatorFile(filename, 0, propagator, "<c8", sublatt_size)


def fromPropagatorFileFast(filename: str, offset: int, dtype: str, sublatt_size: List[int]):
    Lx, Ly, Lz, Lt = sublatt_size

    propagator_raw = readMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2))

    return propagator_raw


def toPropagatorFileFast(
    filename: str, offset: int, propagator_raw: numpy.ndarray, dtype: str, sublatt_size: List[int]
):
    Lx, Ly, Lz, Lt = sublatt_size

    writeMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2), propagator_raw)


def readPropagatorFast(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    sublatt_size = getSublatticeSize(latt_size)

    propagator_raw = fromPropagatorFileFast(filename, 0, "<c8", sublatt_size)
    return propagator_raw


def writePropagatorFast(filename: str, propagator: numpy.ndarray, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    sublatt_size = getSublatticeSize(latt_size)

    toPropagatorFileFast(filename, 0, propagator, "<c8", sublatt_size)
