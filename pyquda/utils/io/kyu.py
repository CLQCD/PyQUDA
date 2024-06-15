from os import path
from typing import List

import numpy

from ... import getSublatticeSize, readMPIFile, writeMPIFile

Nd, Ns, Nc = 4, 4, 3


def fromGaugeFile(filename: str, offset: int, dtype: str, sublatt_size: List[int]):
    Lx, Ly, Lz, Lt = sublatt_size

    gauge_raw = readMPIFile(filename, dtype, offset, (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx), (7, 6, 5, 4))
    gauge_raw = (
        gauge_raw.transpose(0, 4, 5, 6, 7, 2, 1, 3)
        .astype("<f8")
        .copy()
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2)
        .view("<c16")
    )

    return gauge_raw


def toGaugeFile(filename: str, offset: int, gauge_raw: numpy.ndarray, dtype: str, sublatt_size: List[int]):
    Lx, Ly, Lz, Lt = sublatt_size

    gauge_raw = (
        gauge_raw.view("<f8")
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc, 2)
        .astype(dtype)
        .transpose(0, 6, 5, 7, 1, 2, 3, 4)
        .copy()
    )
    writeMPIFile(filename, dtype, offset, (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx), (7, 6, 5, 4), gauge_raw)


def readGauge(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    sublatt_size = getSublatticeSize(latt_size)
    gauge_raw = fromGaugeFile(filename, 0, ">f8", sublatt_size)
    return gauge_raw


def writeGauge(filename: str, gauge_raw: numpy.ndarray, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    sublatt_size = getSublatticeSize(latt_size)
    toGaugeFile(filename, 0, gauge_raw, ">f8", sublatt_size)


def fromPropagatorFile(filename: str, offset: int, dtype: str, sublatt_size: List[int]):
    Lx, Ly, Lz, Lt = sublatt_size

    propagator_raw = readMPIFile(filename, dtype, offset, (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx), (8, 7, 6, 5))
    propagator_raw = (
        propagator_raw.transpose(5, 6, 7, 8, 3, 0, 4, 1, 2)
        .astype("<f8")
        .copy()
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc * 2)
        .view("<c16")
    )

    return propagator_raw


def toPropagatorFile(filename: str, offset: int, propagator_raw: numpy.ndarray, dtype: str, sublatt_size: List[int]):
    Lx, Ly, Lz, Lt = sublatt_size

    propagator_raw = (
        propagator_raw.view("<f8")
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc, 2)
        .astype(dtype)
        .transpose(5, 7, 8, 4, 6, 0, 1, 2, 3)
        .copy()
    )
    writeMPIFile(filename, dtype, offset, (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx), (8, 7, 6, 5), propagator_raw)


def readPropagator(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    sublatt_size = getSublatticeSize(latt_size)
    propagator_raw = fromPropagatorFile(filename, 0, ">f8", sublatt_size)
    return propagator_raw


def writePropagator(filename: str, propagator: numpy.ndarray, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    sublatt_size = getSublatticeSize(latt_size)

    toPropagatorFile(filename, 0, propagator, ">f8", sublatt_size)
