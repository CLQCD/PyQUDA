from typing import List, Union

import numpy

from pyquda.field import Ns, Nc, LatticeInfo, LatticeGauge, LatticePropagator, LatticeStaggeredPropagator, cb2, lexico

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
        propagator.latt_info, contract("ij,wtzyxjkab,kl->wtzyxilab", P, propagator.data, Pinv, optimize=True)
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
        propagator.latt_info, contract("ij,wtzyxjkab,kl->wtzyxilab", P, propagator.data, Pinv, optimize=True)
    )


def readChromaQIOGauge(filename: str, checksum: bool = True):
    from .chroma import readQIOGauge as read

    latt_size, gauge_raw = read(filename, checksum)
    return LatticeGauge(LatticeInfo(latt_size), cb2(gauge_raw, [1, 2, 3, 4]))


def readILDGBinGauge(filename: str, dtype: str, latt_size: List[int]):
    from .chroma import readILDGBinGauge as read

    gauge_raw = read(filename, dtype, latt_size)
    return LatticeGauge(LatticeInfo(latt_size), cb2(gauge_raw, [1, 2, 3, 4]))


def readChromaQIOPropagator(filename: str, checksum: bool = True):
    from .chroma import readQIOPropagator as read

    latt_size, staggered, propagator_raw = read(filename, checksum)
    if not staggered:
        return LatticePropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3]))
    else:
        return LatticeStaggeredPropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3]))


def readMILCGauge(filename: str, checksum: bool = True):
    from .milc import readGauge as read

    latt_size, gauge_raw = read(filename, checksum)
    return LatticeGauge(LatticeInfo(latt_size), cb2(gauge_raw, [1, 2, 3, 4]))


def writeMILCGauge(filename: str, gauge: LatticeGauge):
    from .milc import writeGauge as write

    write(filename, gauge.latt_info.global_size, gauge.lexico())


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

    write(filename, gauge.latt_info.global_size, gauge.lexico())


def readKYUPropagator(filename: str, latt_size: List[int]):
    from .kyu import readPropagator as read

    propagator_raw = read(filename, latt_size)
    return rotateToDeGrandRossi(LatticePropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3])))


def writeKYUPropagator(filename: str, propagator: LatticePropagator):
    from .kyu import writePropagator as write

    write(filename, propagator.latt_info.global_size, rotateToDiracPauli(propagator).lexico())


def readXQCDPropagator(filename: str, latt_size: List[int], staggered: bool):
    from .xqcd import readPropagator as read

    propagator_raw = read(filename, latt_size, staggered)
    if not staggered:
        return rotateToDeGrandRossi(LatticePropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3])))
    else:
        return LatticeStaggeredPropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3]))


def writeXQCDPropagator(filename: str, propagator: Union[LatticePropagator, LatticeStaggeredPropagator]):
    from .xqcd import writePropagator as write

    latt_size = propagator.latt_info.global_size
    staggered = isinstance(propagator, LatticeStaggeredPropagator)
    if not staggered:
        write(filename, latt_size, rotateToDiracPauli(propagator).lexico(), staggered)
    else:
        write(filename, latt_size, propagator.lexico(), staggered)


def readXQCDPropagatorFast(filename: str, latt_size: List[int]):
    from .xqcd import readPropagatorFast as read

    latt_info = LatticeInfo(latt_size)
    Lx, Ly, Lz, Lt = latt_info.size
    propagator_raw = read(filename, latt_size)
    propagator = LatticePropagator(latt_info, cb2(propagator_raw, [2, 3, 4, 5]))
    propagator.data = propagator.data.reshape(Ns, Nc, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    propagator.toDevice()
    propagator.data = propagator.data.transpose(2, 3, 4, 5, 6, 7, 0, 8, 1).astype("<c16")

    return rotateToDeGrandRossi(propagator)


def writeXQCDPropagatorFast(filename: str, propagator: LatticePropagator):
    from .xqcd import writePropagatorFast as write

    latt_info = propagator.latt_info
    Lx, Ly, Lz, Lt = latt_info.size
    propagator = rotateToDiracPauli(propagator)
    propagator.data = propagator.data.astype("<c8").transpose(6, 8, 0, 1, 2, 3, 4, 5, 7)
    propagator.toHost()
    propagator.data = propagator.data.reshape(Ns, Nc, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    propagator_raw = lexico(propagator.data, [2, 3, 4, 5, 6])
    write(filename, latt_info.global_size, propagator_raw)


def readNPYGauge(filename: str):
    from .npy import readGauge as read

    filename = filename if filename.endswith(".npy") else filename + ".npy"
    latt_size, gauge_raw = read(filename)
    return LatticeGauge(LatticeInfo(latt_size), cb2(gauge_raw, [1, 2, 3, 4]))


def writeNPYGauge(filename: str, gauge: LatticeGauge):
    from .npy import writeGauge as write

    filename = filename if filename.endswith(".npy") else filename + ".npy"
    write(filename, gauge.latt_info.global_size, gauge.lexico())


def readNPYPropagator(filename: str):
    from .npy import readPropagator as read

    latt_size, propagator_raw = read(filename)
    return LatticePropagator(LatticeInfo(latt_size), cb2(propagator_raw, [0, 1, 2, 3]))


def writeNPYPropagator(filename: str, propagator: LatticePropagator):
    from .npy import writePropagator as write

    write(filename, propagator.latt_info.global_size, propagator.lexico())


def readQIOGauge(filename: str):
    return readChromaQIOGauge(filename)


def readQIOPropagator(filename: str):
    return readChromaQIOPropagator(filename)


def readKYUPropagatorF(filename: str, latt_size: List[int]):
    return readXQCDPropagator(filename, latt_size, False)


def writeKYUPropagatorF(filename: str, propagator: LatticePropagator):
    writeXQCDPropagator(filename, propagator, False)


def readXQCDStaggeredPropagator(filename: str, latt_size: List[int]):
    return readXQCDPropagator(filename, latt_size, True)


def writeXQCDStaggeredPropagator(filename: str, propagator: LatticeStaggeredPropagator):
    writeXQCDPropagator(filename, propagator)


def readKYUPropagatorFFast(filename: str, latt_size: List[int]):
    return readXQCDPropagatorFast(filename, latt_size)


def writeKYUPropagatorFFast(filename: str, propagator: LatticePropagator):
    writeXQCDPropagatorFast(filename, propagator)
