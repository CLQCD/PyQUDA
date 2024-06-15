from typing import List

import numpy

from ...field import Ns, Nc, LatticeInfo, LatticeGauge, LatticePropagator, LatticeStaggeredPropagator, cb2, lexico

from .eigen import readTimeSlice as readTimeSliceEivenvector


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


def readChromaQIOGauge(filename: str):
    from .chroma import readQIOGauge as read

    latt_size, gauge_raw = read(filename)
    return LatticeGauge(LatticeInfo(latt_size), cb2(gauge_raw, [1, 2, 3, 4]))


def readQIOGauge(filename: str):
    return readChromaQIOGauge(filename)


def readILDGBinGauge(filename: str, dtype: str, latt_size: List[int]):
    from .chroma import readILDGBinGauge as read

    gauge_raw = read(filename, dtype, latt_size)
    return LatticeGauge(LatticeInfo(latt_size), cb2(gauge_raw, [1, 2, 3, 4]))


def readChromaQIOPropagator(filename: str):
    from .chroma import readQIOPropagator as read

    latt_size, staggered, propagator_raw = read(filename)
    if not staggered:
        return LatticePropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3]))
    else:
        return LatticeStaggeredPropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3]))


def readQIOPropagator(filename: str):
    return readChromaQIOPropagator(filename)


def readMILCGauge(filename: str):
    from .milc import readGauge as read

    latt_size, gauge_raw = read(filename)
    return LatticeGauge(LatticeInfo(latt_size), cb2(gauge_raw, [1, 2, 3, 4]))


def readMILCQIOPropagator(filename: str):
    from .milc import readQIOPropagator as read

    latt_size, staggered, propagator_raw = read(filename)
    if not staggered:
        return LatticePropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3]))
    else:
        return LatticeStaggeredPropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3]))


def readKYUGauge(filename: str, latt_size: List[int]):
    from .kyu import readGauge as read

    gauge_raw = read(filename, latt_size)
    return LatticeGauge(LatticeInfo(latt_size), cb2(gauge_raw, [1, 2, 3, 4]))


def writeKYUGauge(filename: str, gauge: LatticeGauge):
    from .kyu import writeGauge as write

    write(filename, gauge.lexico(), gauge.latt_info.global_size)


def readKYUPropagator(filename: str, latt_size: List[int]):
    from .kyu import readPropagator as read

    propagator_raw = read(filename, latt_size)
    return rotateToDeGrandRossi(LatticePropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3])))


def writeKYUPropagator(filename: str, propagator: LatticePropagator):
    from .kyu import writePropagator as write

    write(filename, rotateToDiracPauli(propagator).lexico(), propagator.latt_info.global_size)


def readKYUPropagatorF(filename: str, latt_size: List[int]):
    from .kyu_single import readPropagator as read

    propagator_raw = read(filename, latt_size)
    return rotateToDeGrandRossi(LatticePropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3])))


def writeKYUPropagatorF(filename: str, propagator: LatticePropagator):
    from .kyu_single import writePropagator as write

    write(filename, rotateToDiracPauli(propagator).lexico(), propagator.latt_info.global_size)


def readKYUPropagatorFFast(filename: str, latt_size: List[int]):
    from .kyu_single import readPropagatorFast as read

    latt_info = LatticeInfo(latt_size)
    Lx, Ly, Lz, Lt = latt_info.size
    propagator_raw = read(filename, latt_size)
    propagator = LatticePropagator(latt_info, cb2(propagator_raw, [2, 3, 4, 5]))
    propagator.data = propagator.data.reshape(Ns, Nc, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    propagator.toDevice()
    propagator.data = propagator.data.transpose(2, 3, 4, 5, 6, 7, 0, 8, 1).astype("<c16")

    return rotateToDeGrandRossi(propagator)


def writeKYUPropagatorFFast(filename: str, propagator: LatticePropagator):
    from .kyu_single import writePropagatorFast as write

    latt_info = propagator.latt_info
    Lx, Ly, Lz, Lt = latt_info.size
    propagator = rotateToDiracPauli(propagator)
    propagator.data = propagator.data.astype("<c8").transpose(6, 8, 0, 1, 2, 3, 4, 5, 7)
    propagator.toHost()
    propagator.data = propagator.data.reshape(Ns, Nc, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    propagator_raw = lexico(propagator.data, [2, 3, 4, 5, 6])
    write(filename, propagator_raw, latt_info.global_size)
