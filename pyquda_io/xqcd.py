from os import path
from typing import List

import numpy

from ._mpi_file import getSublatticeSize, readMPIFile, writeMPIFile
from ._field_utils import propagatorFromDiracPauli, propagatorToDiracPauli

Ns, Nc = 4, 3


def readPropagator(filename: str, latt_size: List[int], staggered: bool):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", 0

    if not staggered:
        propagator = readMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2))
        propagator = propagator.transpose(2, 3, 4, 5, 6, 0, 7, 1).astype("<c16")
        propagator = propagatorFromDiracPauli(propagator)
    else:
        # QDP_ALIGN16 makes the last Nc to be aligned with 16 Bytes.
        propagator_align16 = readMPIFile(filename, dtype, offset, (Nc, Lt, Lz, Ly, Lx, 4), (4, 3, 2, 1))
        propagator = propagator_align16[:, :, :, :, :, :Nc]
        propagator = propagator.transpose(1, 2, 3, 4, 5, 0).astype("<c16")
    return propagator


def writePropagator(
    filename: str, latt_size: List[int], propagator: numpy.ndarray, staggered: bool
):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", 0

    if not staggered:
        propagator = propagatorToDiracPauli(propagator)
        propagator = propagator.astype(dtype).transpose(5, 7, 0, 1, 2, 3, 4, 6).copy()
        writeMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2), propagator)
    else:
        # QDP_ALIGN16 makes the last Nc to be aligned with 16 Bytes.
        propagator = propagator.astype(dtype).transpose(5, 0, 1, 2, 3, 4)
        propagator_align16 = numpy.zeros((Nc, Lt, Lz, Ly, Lx, 4), dtype)
        propagator_align16[:, :, :, :, :, :Nc] = propagator
        writeMPIFile(filename, dtype, offset, (Nc, Lt, Lz, Ly, Lx, 4), (4, 3, 2, 1), propagator_align16)


def readPropagatorFast(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", 0

    propagator = readMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2))
    return propagator


def writePropagatorFast(filename: str, latt_size: List[int], propagator: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", 0

    writeMPIFile(filename, dtype, offset, (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), (5, 4, 3, 2), propagator)
