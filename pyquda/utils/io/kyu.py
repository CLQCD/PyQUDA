from os import path

import numpy

from ...field import Ns, Nc, Nd, LatticeInfo, LatticeGauge, LatticePropagator, cb2


def fromGaugeBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from ... import readMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    gauge_raw = readMPIFile(
        filename,
        offset,
        dtype,
        (Nd, Nc, Nc, 2, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    gauge_raw = (
        gauge_raw.transpose(0, 4, 5, 6, 7, 2, 1, 3)
        .astype("<f8")
        .copy()
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2)
        .view("<c16")
    )

    return gauge_raw


def toGaugeBuffer(filename: str, offset: int, gauge_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from ... import writeMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    gauge_raw = (
        gauge_raw.view("<f8")
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc, 2)
        .astype(dtype)
        .transpose(0, 6, 5, 7, 1, 2, 3, 4)
        .copy()
    )
    writeMPIFile(
        filename,
        offset,
        gauge_raw,
        dtype,
        (Nd, Nc, Nc, 2, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )


def readGauge(filename: str, latt_info: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    gauge_raw = fromGaugeBuffer(filename, 0, ">f8", latt_info)

    return LatticeGauge(latt_info, cb2(gauge_raw, [1, 2, 3, 4]))


def writeGauge(filename: str, gauge: LatticeGauge):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = gauge.latt_info

    toGaugeBuffer(filename, 0, gauge.lexico(), ">f8", latt_info)


# matrices to convert gamma basis bewteen DeGrand-Rossi and Dirac-Pauli
# \psi(DP) = _DR_TO_DP \psi(DR)
# \psi(DR) = _DP_TO_DR \psi(DP)
_DP_TO_DR = numpy.array(
    [
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 1],
        [-1, 0, -1, 0],
    ]
)
_DR_TO_DP = numpy.array(
    [
        [0, -1, 0, -1],
        [1, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
    ]
)


def rotateToDiracPauli(propagator: LatticePropagator):
    from opt_einsum import contract

    if propagator.location == "numpy":
        P = numpy.asarray(_DR_TO_DP)
        Pinv = numpy.asarray(_DP_TO_DR) / 2
    elif propagator.location == "cupy":
        import cupy

        P = cupy.asarray(_DR_TO_DP)
        Pinv = cupy.asarray(_DP_TO_DR) / 2
    elif propagator.location == "torch":
        import torch

        P = torch.as_tensor(_DR_TO_DP)
        Pinv = torch.as_tensor(_DP_TO_DR) / 2

    return LatticePropagator(
        propagator.latt_info, contract("ij,etzyxjkab,kl->etzyxilab", P, propagator.data, Pinv, optimize=True)
    )


def rotateToDeGrandRossi(propagator: LatticePropagator):
    from opt_einsum import contract

    if propagator.location == "numpy":
        P = numpy.asarray(_DP_TO_DR)
        Pinv = numpy.asarray(_DR_TO_DP) / 2
    elif propagator.location == "cupy":
        import cupy

        P = cupy.asarray(_DP_TO_DR)
        Pinv = cupy.asarray(_DR_TO_DP) / 2
    elif propagator.location == "torch":
        import torch

        P = torch.as_tensor(_DP_TO_DR)
        Pinv = torch.as_tensor(_DR_TO_DP) / 2

    return LatticePropagator(
        propagator.latt_info, contract("ij,etzyxjkab,kl->etzyxilab", P, propagator.data, Pinv, optimize=True)
    )


def fromPropagatorBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from ... import readMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_raw = readMPIFile(
        filename,
        offset,
        dtype,
        (Ns, Nc, 2, Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    propagator_raw = (
        propagator_raw.transpose(5, 6, 7, 8, 3, 0, 4, 1, 2)
        .astype("<f8")
        .copy()
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc * 2)
        .view("<c16")
    )

    return propagator_raw


def toPropagatorBuffer(filename: str, offset: int, propagator_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from ... import writeMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_raw = (
        propagator_raw.view("<f8")
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc, 2)
        .astype(dtype)
        .transpose(5, 7, 8, 4, 6, 0, 1, 2, 3)
        .copy()
    )
    writeMPIFile(
        filename,
        offset,
        propagator_raw,
        dtype,
        (Ns, Nc, 2, Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )


def readPropagator(filename: str, latt_info: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    propagator_raw = fromPropagatorBuffer(filename, 0, ">f8", latt_info)

    return rotateToDeGrandRossi(LatticePropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3])))


def writePropagator(filename: str, propagator: LatticePropagator):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = propagator.latt_info

    toPropagatorBuffer(filename, 0, rotateToDiracPauli(propagator).lexico(), ">f8", latt_info)
