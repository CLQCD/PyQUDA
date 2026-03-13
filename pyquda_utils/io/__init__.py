from typing import List

import numpy

from pyquda.field import LatticeInfo, LatticeGauge, LatticePropagator, LatticeStaggeredPropagator, evenodd, lexico

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


def readChromaQIOGauge(filename: str, checksum: bool = True, reunitarize_sigma: float = 1e-6):
    from pyquda_io.chroma import readQIOGauge as read

    latt_size, gauge_raw = read(filename, checksum, reunitarize_sigma)
    latt_info = LatticeInfo(latt_size)
    return LatticeGauge(latt_info, latt_info.evenodd(gauge_raw, True))


def readILDGGauge(filename: str, checksum: bool = True, reunitarize_sigma: float = 1e-6):
    from pyquda_io.ildg import readGauge as read

    latt_size, gauge_raw = read(filename, checksum, reunitarize_sigma)
    latt_info = LatticeInfo(latt_size)
    return LatticeGauge(latt_info, latt_info.evenodd(gauge_raw, True))


def readILDGBinGauge(filename: str, dtype: str, latt_size: List[int]):
    from pyquda_io.ildg import readBinGauge as read

    gauge_raw = read(filename, dtype, latt_size)
    latt_info = LatticeInfo(latt_size)
    return LatticeGauge(latt_info, latt_info.evenodd(gauge_raw, True))


def readChromaQIOPropagator(filename: str, checksum: bool = True):
    from pyquda_io.chroma import readQIOPropagator as read

    latt_size, propagator_raw = read(filename, checksum)
    latt_info = LatticeInfo(latt_size)
    return LatticePropagator(latt_info, latt_info.evenodd(propagator_raw, False))


def readChromaQIOStaggeredPropagator(filename: str, checksum: bool = True):
    from pyquda_io.chroma import readQIOStaggeredPropagator as read

    latt_size, propagator_raw = read(filename, checksum)
    latt_info = LatticeInfo(latt_size)
    return LatticeStaggeredPropagator(latt_info, latt_info.evenodd(propagator_raw, False))


def readMILCGauge(filename: str, checksum: bool = True, reunitarize_sigma: float = 1e-6):
    from pyquda_io.milc import readGauge as read

    latt_size, gauge_raw = read(filename, checksum, reunitarize_sigma)
    latt_info = LatticeInfo(latt_size)
    return LatticeGauge(latt_info, latt_info.evenodd(gauge_raw, True))


def writeMILCGauge(filename: str, gauge: LatticeGauge):
    from pyquda_io.milc import writeGauge as write

    write(filename, gauge.latt_info.global_size, gauge.lexico())


def readMILCQIOPropagator(filename: str):
    from pyquda_io.milc import readQIOPropagator as read

    latt_size, propagator_raw = read(filename)
    latt_info = LatticeInfo(latt_size)
    return LatticePropagator(latt_info, latt_info.evenodd(propagator_raw, False))


def readMILCQIOStaggeredPropagator(filename: str):
    from pyquda_io.milc import readQIOStaggeredPropagator as read

    latt_size, propagator_raw = read(filename)
    latt_info = LatticeInfo(latt_size)
    return LatticeStaggeredPropagator(latt_info, latt_info.evenodd(propagator_raw, False))


def readKYUGauge(filename: str, latt_size: List[int]):
    from pyquda_io.kyu import readGauge as read

    gauge_raw = read(filename, latt_size)
    latt_info = LatticeInfo(latt_size)
    return LatticeGauge(latt_info, latt_info.evenodd(gauge_raw, True))


def writeKYUGauge(filename: str, gauge: LatticeGauge):
    from pyquda_io.kyu import writeGauge as write

    write(filename, gauge.latt_info.global_size, gauge.lexico())


def readKYUPropagator(filename: str, latt_size: List[int]):
    from pyquda_io.kyu import readPropagator as read

    propagator_raw = read(filename, latt_size)
    latt_info = LatticeInfo(latt_size)
    return LatticePropagator(latt_info, latt_info.evenodd(propagator_raw, False))


def writeKYUPropagator(filename: str, propagator: LatticePropagator):
    from pyquda_io.kyu import writePropagator as write

    write(filename, propagator.latt_info.global_size, propagator.lexico())


def readXQCDPropagator(filename: str, latt_size: List[int]):
    from pyquda_io.xqcd import readPropagator as read

    propagator_raw = read(filename, latt_size)
    latt_info = LatticeInfo(latt_size)
    return LatticePropagator(latt_info, latt_info.evenodd(propagator_raw, False))


def readXQCDStaggeredPropagator(filename: str, latt_size: List[int]):
    from pyquda_io.xqcd import readStaggeredPropagator as read

    propagator_raw = read(filename, latt_size)
    latt_info = LatticeInfo(latt_size)
    return LatticeStaggeredPropagator(latt_info, latt_info.evenodd(propagator_raw, False))


def writeXQCDPropagator(filename: str, propagator: LatticePropagator):
    from pyquda_io.xqcd import writePropagator as write

    write(filename, propagator.latt_info.global_size, propagator.lexico())


def writeXQCDStaggeredPropagator(filename: str, propagator: LatticeStaggeredPropagator):
    from pyquda_io.xqcd import writeStaggeredPropagator as write

    write(filename, propagator.latt_info.global_size, propagator.lexico())


def readNPYGauge(filename: str):
    from pyquda_io.npy import readGauge as read

    filename = filename if filename.endswith(".npy") else filename + ".npy"
    latt_size, gauge_raw = read(filename)
    latt_info = LatticeInfo(latt_size)
    return LatticeGauge(latt_info, latt_info.evenodd(gauge_raw, True))


def writeNPYGauge(filename: str, gauge: LatticeGauge):
    from pyquda_io.npy import writeGauge as write

    filename = filename if filename.endswith(".npy") else filename + ".npy"
    write(filename, gauge.latt_info.global_size, gauge.lexico())


def readNPYPropagator(filename: str):
    from pyquda_io.npy import readPropagator as read

    latt_size, propagator_raw = read(filename)
    latt_info = LatticeInfo(latt_size)
    return LatticePropagator(latt_info, latt_info.evenodd(propagator_raw, False))


def writeNPYPropagator(filename: str, propagator: LatticePropagator):
    from pyquda_io.npy import writePropagator as write

    write(filename, propagator.latt_info.global_size, propagator.lexico())


def readOpenQCDGauge(filename: str, plaquette: bool = True):
    from pyquda_io.openqcd import readGauge as read

    latt_size, gauge = read(filename, plaquette, False)
    latt_info = LatticeInfo(latt_size)
    return LatticeGauge(latt_info, gauge)


def writeOpenQCDGauge(filename: str, gauge: LatticeGauge):
    from pyquda_io.openqcd import writeGauge as write

    write(filename, gauge.latt_info.global_size, gauge.getHost(), False)


def readNERSCGauge(
    filename: str,
    checksum: bool = True,
    plaquette: bool = True,
    link_trace: bool = True,
    reunitarize_sigma: float = 1e-6,
):
    from pyquda_io.nersc import readGauge as read

    latt_size, gauge_raw = read(filename, checksum, plaquette, link_trace, reunitarize_sigma)
    latt_info = LatticeInfo(latt_size)
    return LatticeGauge(latt_info, latt_info.evenodd(gauge_raw, True))


def writeNERSCGauge(
    filename: str,
    gauge: LatticeGauge,
    use_fp32: bool = False,
    ensemble_id: str = "PyQUDA",
    ensemble_label: str = "",
    sequence_number: int = 0,
):
    from pyquda_io.nersc import writeGauge as write

    write(filename, gauge.latt_info.global_size, gauge.lexico(), use_fp32, ensemble_id, ensemble_label, sequence_number)


def readQIOGauge(filename: str):
    return readChromaQIOGauge(filename)


def readQIOPropagator(filename: str):
    return readChromaQIOPropagator(filename)


def readQIOStaggeredPropagator(filename: str):
    return readChromaQIOStaggeredPropagator(filename)


def readKYUPropagatorF(filename: str, latt_size: List[int]):
    return readXQCDPropagator(filename, latt_size)


def writeKYUPropagatorF(filename: str, propagator: LatticePropagator):
    writeXQCDPropagator(filename, propagator)


from pyquda_comm.array import arrayDevice


def rotateToDiracPauli(propagator: LatticePropagator):
    from opt_einsum import contract

    location = propagator.location
    P = arrayDevice(_DR_TO_DP, location)
    Pinv = arrayDevice(_DP_TO_DR, location) / 2

    return LatticePropagator(
        propagator.latt_info, contract("ij,wtzyxjkab,kl->wtzyxilab", P, propagator.data, Pinv, optimize=True)
    )


def rotateToDeGrandRossi(propagator: LatticePropagator):
    from opt_einsum import contract

    location = propagator.location
    P = arrayDevice(_DP_TO_DR, location)
    Pinv = arrayDevice(_DR_TO_DP, location) / 2

    return LatticePropagator(
        propagator.latt_info, contract("ij,wtzyxjkab,kl->wtzyxilab", P, propagator.data, Pinv, optimize=True)
    )


def readXQCDPropagatorFast(filename: str, latt_size: List[int]):
    from pyquda_comm import getArrayBackend
    from pyquda_comm.array import arrayDevice
    from pyquda_io.xqcd import readPropagatorFast as read

    latt_info = LatticeInfo(latt_size)
    propagator_raw = read(filename, latt_size)
    propagator = LatticePropagator(
        latt_info,
        arrayDevice(evenodd(propagator_raw, [2, 3, 4, 5]), getArrayBackend())
        .transpose(2, 3, 4, 5, 6, 7, 0, 8, 1)
        .astype("<c16"),
    )
    # Ns, Nc = 4, 3
    # Lx, Ly, Lz, Lt = latt_info.size
    # propagator = LatticePropagator(latt_info, evenodd(propagator_raw, [2, 3, 4, 5]))
    # propagator.data = propagator.data.reshape(Ns, Nc, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    # propagator.toDevice()
    # propagator.data = propagator.data.transpose(2, 3, 4, 5, 6, 7, 0, 8, 1).astype("<c16")

    return rotateToDeGrandRossi(propagator)


def writeXQCDPropagatorFast(filename: str, propagator: LatticePropagator):
    from pyquda_comm import getArrayBackend
    from pyquda_comm.array import arrayHost
    from pyquda_io.xqcd import writePropagatorFast as write

    latt_info = propagator.latt_info
    propagator = rotateToDiracPauli(propagator)
    propagator_raw = lexico(
        arrayHost(propagator.data.transpose(6, 8, 0, 1, 2, 3, 4, 5, 7).astype("<c8"), getArrayBackend()),
        [2, 3, 4, 5, 6],
    )
    # Ns, Nc = 4, 3
    # Lx, Ly, Lz, Lt = latt_info.size
    # propagator.data = propagator.data.astype("<c8").transpose(6, 8, 0, 1, 2, 3, 4, 5, 7)
    # propagator.toHost()
    # propagator.data = propagator.data.reshape(Ns, Nc, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    # propagator_raw = lexico(propagator.data, [2, 3, 4, 5, 6])
    write(filename, latt_info.global_size, propagator_raw)


def readKYUPropagatorFFast(filename: str, latt_size: List[int]):
    return readXQCDPropagatorFast(filename, latt_size)


def writeKYUPropagatorFFast(filename: str, propagator: LatticePropagator):
    writeXQCDPropagatorFast(filename, propagator)
