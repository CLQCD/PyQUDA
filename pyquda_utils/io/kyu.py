from os import path
from typing import List

import numpy

from .mpi_file import getSublatticeSize, readMPIFile, writeMPIFile

Nd, Ns, Nc = 4, 4, 3


def readGauge(filename: str, latt_size: List[int], grid_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    dtype, offset = ">f8", 0

    gauge = readMPIFile(filename, dtype, offset, (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx), (7, 6, 5, 4), grid_size)
    gauge = (
        gauge.transpose(0, 4, 5, 6, 7, 2, 1, 3)
        .astype("<f8")
        .copy()
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2)
        .view("<c16")
    )
    return gauge


def writeGauge(filename: str, latt_size: List[int], grid_size: List[int], gauge: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    dtype, offset = ">f8", 0

    gauge = (
        gauge.view("<f8").reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc, 2).astype(dtype).transpose(0, 6, 5, 7, 1, 2, 3, 4).copy()
    )
    writeMPIFile(filename, dtype, offset, (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx), (7, 6, 5, 4), grid_size, gauge)


def readPropagator(filename: str, latt_size: List[int], grid_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    dtype, offset = ">f8", 0

    propagator = readMPIFile(filename, dtype, offset, (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx), (8, 7, 6, 5), grid_size)
    propagator = (
        propagator.transpose(5, 6, 7, 8, 3, 0, 4, 1, 2)
        .astype("<f8")
        .copy()
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc * 2)
        .view("<c16")
    )
    return propagator


def writePropagator(filename: str, latt_size: List[int], grid_size: List[int], propagator: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    dtype, offset = ">f8", 0

    propagator = (
        propagator.view("<f8")
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc, 2)
        .astype(dtype)
        .transpose(5, 7, 8, 4, 6, 0, 1, 2, 3)
        .copy()
    )
    writeMPIFile(filename, dtype, offset, (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx), (8, 7, 6, 5), grid_size, propagator)
