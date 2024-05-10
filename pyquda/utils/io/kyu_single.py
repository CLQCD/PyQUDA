from os import path

import numpy

from ...field import Ns, Nc, LatticeInfo, LatticePropagator, cb2


# matrices to convert gamma basis bewteen DeGrand-Rossi and Dirac-Pauli
# \psi(DP) = _DR_TO_DP \psi(DR)
# \psi(DR) = _DP_TO_DR \psi(DP)
_DP_TO_DR = [
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
    [0, 1, 0, 1],
    [-1, 0, -1, 0],
]
_DR_TO_DP = [
    [0, -1, 0, -1],
    [1, 0, 1, 0],
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
]


def rotateToDiracPauli(propagator: LatticePropagator):
    from opt_einsum import contract

    if propagator.location == "numpy":
        A = numpy.asarray(_DP_TO_DR)
        Ainv = numpy.asarray(_DR_TO_DP)
    elif propagator.location == "cupy":
        import cupy

        A = cupy.asarray(_DP_TO_DR)
        Ainv = cupy.asarray(_DR_TO_DP)
    elif propagator.location == "torch":
        import torch

        A = torch.as_tensor(_DP_TO_DR)
        Ainv = torch.as_tensor(_DR_TO_DP)

    return LatticePropagator(
        propagator.latt_info, contract("ij,etzyxjkab,kl->etzyxilab", Ainv, propagator.data, A, optimize=True) / 2
    )


def rotateToDeGrandRossi(propagator: LatticePropagator):
    from opt_einsum import contract

    if propagator.location == "numpy":
        A = numpy.asarray(_DR_TO_DP)
        Ainv = numpy.asarray(_DP_TO_DR)
    elif propagator.location == "cupy":
        import cupy

        A = cupy.array(_DR_TO_DP)
        Ainv = cupy.array(_DP_TO_DR)
    elif propagator.location == "torch":
        import torch

        A = torch.as_tensor(_DR_TO_DP)
        Ainv = torch.as_tensor(_DP_TO_DR)

    return LatticePropagator(
        propagator.latt_info, contract("ij,etzyxjkab,kl->etzyxilab", Ainv, propagator.data, A, optimize=True) / 2
    )


def fromPropagatorBuffer(buffer: bytes, dtype: str, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_raw = (
        numpy.frombuffer(buffer, dtype)
        .reshape(Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc)[
            :,
            :,
            gt * Lt : (gt + 1) * Lt,
            gz * Lz : (gz + 1) * Lz,
            gy * Ly : (gy + 1) * Ly,
            gx * Lx : (gx + 1) * Lx,
        ]
        .transpose(2, 3, 4, 5, 6, 0, 7, 1)
        .astype("<c16")
    )

    return propagator_raw


def toPropagatorBuffer(propagator_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from .gather_raw import gatherPropagatorRaw

    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_gathered_raw = gatherPropagatorRaw(propagator_raw, latt_info)
    if latt_info.mpi_rank == 0:
        buffer = (
            propagator_gathered_raw.astype(dtype)
            .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Ns, Nc, Nc)
            .transpose(5, 7, 0, 1, 2, 3, 4, 6)
            .copy()
            .tobytes()
        )
    else:
        buffer = None

    return buffer


def readPropagator(filename: str, latt_info: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        # kyu_binary_data = f.read(2 * Ns * Nc * Lt * Lz * Ly * Lx * 8)
        kyu_binary_data = f.read()
    propagator_raw = fromPropagatorBuffer(kyu_binary_data, "<c8", latt_info)

    return rotateToDeGrandRossi(LatticePropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3])))


def writePropagator(filename: str, propagator: LatticePropagator):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = propagator.latt_info

    kyu_binary_data = toPropagatorBuffer(rotateToDiracPauli(propagator).lexico(), "<c8", latt_info)
    if latt_info.mpi_rank == 0:
        with open(filename, "wb") as f:
            f.write(kyu_binary_data)
