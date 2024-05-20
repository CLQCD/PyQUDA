from os import path
from typing import List, Union

import numpy

from ...field import Ns, Nc, LatticeInfo, LatticePropagator, cb2

from .kyu import rotateToDiracPauli, rotateToDeGrandRossi


def fromPropagatorBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from ... import readMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_raw = readMPIFile(
        filename,
        offset,
        dtype,
        (Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc),
        (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc),
        (0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
    )
    propagator_raw = propagator_raw.transpose(2, 3, 4, 5, 6, 0, 7, 1).astype("<c16")

    return propagator_raw


def toPropagatorBuffer(filename: str, offset: int, propagator_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from ... import writeMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_raw = propagator_raw.astype(dtype).transpose(5, 7, 0, 1, 2, 3, 4, 6).copy()
    writeMPIFile(
        filename,
        offset,
        propagator_raw,
        dtype,
        (Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc),
        (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc),
        (0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
    )


def readPropagator(filename: str, latt_info: Union[LatticeInfo, List[int]]):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = LatticeInfo(latt_info) if not isinstance(latt_info, LatticeInfo) else latt_info
    propagator_raw = fromPropagatorBuffer(filename, 0, "<c8", latt_info)

    return rotateToDeGrandRossi(LatticePropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3])))


def writePropagator(filename: str, propagator: LatticePropagator):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = propagator.latt_info

    toPropagatorBuffer(filename, 0, rotateToDiracPauli(propagator).lexico(), "<c8", latt_info)


def fromPropagatorBufferFast(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from ... import readMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_raw = readMPIFile(
        filename,
        offset,
        dtype,
        (Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc),
        (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc),
        (0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
    )
    propagator = LatticePropagator(latt_info, cb2(propagator_raw, [2, 3, 4, 5]))
    propagator.data = propagator.data.reshape(Ns, Nc, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    propagator.toDevice()
    propagator.data = propagator.data.transpose(2, 3, 4, 5, 6, 7, 0, 8, 1).astype("<c16")

    return propagator


def toPropagatorBufferFast(
    filename: str, offset: int, propagator: LatticePropagator, dtype: str, latt_info: LatticeInfo
):
    from ... import writeMPIFile
    from ...field import lexico

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator.data = propagator.data.astype("<c8").transpose(6, 8, 0, 1, 2, 3, 4, 5, 7)
    propagator.toHost()
    propagator.data = propagator.data.reshape(Ns, Nc, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    propagator_raw = lexico(propagator.data, [2, 3, 4, 5, 6])
    writeMPIFile(
        filename,
        offset,
        propagator_raw,
        dtype,
        (Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc),
        (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc),
        (0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
    )


def readPropagatorFast(filename: str, latt_info: Union[LatticeInfo, List[int]]):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = LatticeInfo(latt_info) if not isinstance(latt_info, LatticeInfo) else latt_info

    return rotateToDeGrandRossi(fromPropagatorBufferFast(filename, 0, "<c8", latt_info))


def writePropagatorFast(filename: str, propagator: LatticePropagator):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = propagator.latt_info

    toPropagatorBufferFast(filename, 0, rotateToDiracPauli(propagator), "<c8", latt_info)
