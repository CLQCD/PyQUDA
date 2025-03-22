from os import path
from time import perf_counter
from typing import Sequence

import numpy as np

from pyquda_comm import getLogger, readMPIFile
from pyquda.field import LatticeInfo, LatticeGauge, MultiLatticeFermion, evenodd
from . import pygwu


def init():
    import atexit

    pygwu.init()
    atexit.register(pygwu.shutdown)


# matrices to convert gamma basis bewteen DeGrand-Rossi and Dirac-Pauli
# DP for Dirac-Pauli, DR for DeGrand-Rossi
# \psi(DP) = _DR_TO_DP \psi(DR)
# \psi(DR) = _DP_TO_DR \psi(DP)
_FROM_DIRAC_PAULI = np.array(
    [
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 1],
        [-1, 0, -1, 0],
    ],
    "<i4",
)
_TO_DIRAC_PAULI = np.array(
    [
        [0, -1, 0, -1],
        [1, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
    ],
    "<i4",
)


def multiFermionFromDiracPauli(dirac_pauli: MultiLatticeFermion):
    P = _FROM_DIRAC_PAULI
    N = dirac_pauli.L5 // 12
    dirac_pauli_data = dirac_pauli.data.reshape(N, 4, 3, -1, 4, 3) / 2
    degrand_rossi = MultiLatticeFermion(dirac_pauli.latt_info, dirac_pauli.L5)
    degrand_rossi_data = degrand_rossi.data.reshape(N, 4, 3, -1, 4, 3)
    for i in range(4):
        for j in range(4):
            for _i in range(4):
                for _j in range(4):
                    if P[i, _i] * P[j, _j] == 1:
                        degrand_rossi_data[:, j, :, :, i, :] += dirac_pauli_data[:, _j, :, :, _i, :]
                    if P[i, _i] * P[j, _j] == -1:
                        degrand_rossi_data[:, j, :, :, i, :] -= dirac_pauli_data[:, _j, :, :, _i, :]
    return degrand_rossi


def multiFermionToDiracPauli(degrand_rossi: MultiLatticeFermion):
    P = _TO_DIRAC_PAULI
    N = degrand_rossi.L5 // 12
    degrand_rossi_data = degrand_rossi.data.reshape(N, 4, 3, -1, 4, 3) / 2
    dirac_pauli = MultiLatticeFermion(degrand_rossi.latt_info, degrand_rossi.L5)
    dirac_pauli_data = dirac_pauli.data.reshape(N, 4, 3, -1, 4, 3)
    for i in range(4):
        for j in range(4):
            for _i in range(4):
                for _j in range(4):
                    if P[i, _i] * P[j, _j] == 1:
                        dirac_pauli_data[:, j, :, :, i, :] += degrand_rossi_data[:, _j, :, :, _i, :]
                    if P[i, _i] * P[j, _j] == -1:
                        dirac_pauli_data[:, j, :, :, i, :] -= degrand_rossi_data[:, _j, :, :, _i, :]
    return dirac_pauli


_MINUS_FROM_DIRAC_PAULI = np.array(
    [
        [0, -1, 0, 1],
        [1, 0, -1, 0],
        [0, 1, 0, 1],
        [-1, 0, -1, 0],
    ],
    "<i4",
)
_MINUS_TO_DIRAC_PAULI = np.array(
    [
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, -1, 0, -1],
        [1, 0, 1, 0],
    ],
    "<i4",
)


def eigenVectorFromDiracPauli(dirac_pauli: MultiLatticeFermion):
    """Convert to the negative DeGrand-Rossi basis"""
    data = dirac_pauli.data / 2**0.5
    degrand_rossi = MultiLatticeFermion(dirac_pauli.latt_info, dirac_pauli.L5)
    degrand_rossi.data[:, :, :, :, :, :, 0, :] = -data[:, :, :, :, :, :, 1] + data[:, :, :, :, :, :, 3]
    degrand_rossi.data[:, :, :, :, :, :, 1, :] = data[:, :, :, :, :, :, 0] - data[:, :, :, :, :, :, 2]
    degrand_rossi.data[:, :, :, :, :, :, 2, :] = data[:, :, :, :, :, :, 1] + data[:, :, :, :, :, :, 3]
    degrand_rossi.data[:, :, :, :, :, :, 3, :] = -data[:, :, :, :, :, :, 0] - data[:, :, :, :, :, :, 2]
    return degrand_rossi


def loadEigenValue(file: str):
    eigvals = []
    with open(file, "r") as f:
        for line in f.readlines():
            if line.startswith("EIGV"):
                tag, real, imag, res = line.strip().split()
                eigvals.append(float(real) + 1j * float(imag))
    return eigvals


def loadHWilsonEigenSystem(latt_info: LatticeInfo, file: str, use_fp32: bool):
    """Convert to the negative DeGrand-Rossi"""
    s = perf_counter()
    file = path.expanduser(path.expandvars(file))
    eigvals = loadEigenValue(f"{file}.eigvals")
    eignum = len(eigvals)
    Lx, Ly, Lz, Lt = latt_info.size
    Ns, Nc = latt_info.Ns, latt_info.Nc
    if use_fp32:
        eigvecs_raw = readMPIFile(f"{file}.s", "<c8", 0, (eignum, Lt, Lz, Ly, Lx, Ns, Nc), (4, 3, 2, 1)).astype("<c16")
    else:
        eigvecs_raw = (
            readMPIFile(file, ">f8", 0, (eignum, 2, Ns, Nc, Lt, Lz, Ly, Lx), (7, 6, 5, 4))
            .astype("<f8")
            .transpose(0, 4, 5, 6, 7, 2, 3, 1)
            .reshape(eignum, Lt, Lz, Ly, Lx, Ns, Nc * 2)
            .copy()
            .view("<c16")
        )
    eigvecs = MultiLatticeFermion(latt_info, eignum, evenodd(eigvecs_raw, [1, 2, 3, 4]))
    eigvecs = eigenVectorFromDiracPauli(eigvecs)
    getLogger().info(f"{perf_counter() - s:.3} secs")
    return eigvals, eigvecs


def loadOverlapEigenSystem(latt_info: LatticeInfo, file: str, use_fp32: bool):
    """Keep the Dirac-Pauli basis"""
    s = perf_counter()
    file = path.expanduser(path.expandvars(file))
    eigvals = loadEigenValue(f"{file}.eigvals")
    eignum = len(eigvals)
    Lx, Ly, Lz, Lt = latt_info.size
    Ns, Nc = latt_info.Ns, latt_info.Nc
    if use_fp32:
        eigvecs_raw = readMPIFile(f"{file}.s", "<c8", 0, (eignum, Lt, Lz, Ly, Lx, Ns, Nc), (4, 3, 2, 1)).astype("<c16")
    else:
        eigvecs_raw = (
            readMPIFile(file, ">f8", 0, (eignum, 2, Ns, Nc, Lt, Lz, Ly, Lx), (7, 6, 5, 4))
            .astype("<f8")
            .transpose(0, 4, 5, 6, 7, 2, 3, 1)
            .reshape(eignum, Lt, Lz, Ly, Lx, Ns, Nc * 2)
            .copy()
            .view("<c16")
        )
    eigvecs = MultiLatticeFermion(latt_info, eignum, evenodd(eigvecs_raw, [1, 2, 3, 4]))
    eigvecs = eigenVectorFromDiracPauli(eigvecs)
    getLogger().info(f"{perf_counter() - s:.3} secs")
    return eigvals, eigvecs


def invertOverlap(
    b: MultiLatticeFermion,
    gauge: LatticeGauge,
    kappa: float,
    masses: Sequence[float],
    tol: float,
    maxiter: int,
    ov_ploy_prec: float,
    ov_use_fp32: float,
    ov_test: int,
    one_minus_half_d: int,
    hw_eigvals: Sequence[complex],
    hw_eigvecs: MultiLatticeFermion,
    hw_eigprec: float,
    ov_eigvals: Sequence[complex],
    ov_eigvecs: MultiLatticeFermion,
    ov_eigprec: float,
    dirac_pauli: bool = False,
):
    latt_info = b.latt_info
    x = MultiLatticeFermion(latt_info, len(masses) * b.L5)
    if not dirac_pauli:
        b = multiFermionToDiracPauli(b)
    pygwu.invert_overlap(
        x.data_ptrs,
        b.data_ptrs,
        gauge.data_ptr(0),
        kappa,
        np.asarray(latt_info.global_size, "<i4"),
        np.asarray(masses, "<f8"),
        tol,
        maxiter,
        ov_ploy_prec,
        ov_use_fp32,
        ov_test,
        one_minus_half_d,
        np.asarray(hw_eigvals, "<c16"),
        hw_eigvecs.data_ptrs,
        hw_eigprec,
        np.asarray(ov_eigvals, "<c16"),
        ov_eigvecs.data_ptrs,
        ov_eigprec,
    )
    if not dirac_pauli:
        x = multiFermionFromDiracPauli(x)
    return x


class Overlap:
    def __init__(self, latt_info: LatticeInfo, gauge: LatticeGauge, kappa: float):
        self.latt_info = latt_info
        self.gauge = gauge
        self.kappa = kappa
        self.hw_eigvals, self.hw_eigvecs = None, None
        self.ov_eigvals, self.ov_eigvecs = None, None

    def loadHWilsonEigenSystem(self, file: str, use_fp32: bool, eigprec: float):
        self.hw_eigvals, self.hw_eigvecs = loadHWilsonEigenSystem(self.latt_info, file, use_fp32)
        self.hw_eigprec = eigprec

    def loadOverlapEigenSystem(self, file: str, use_fp32: bool, eigprec: float):
        self.ov_eigvals, self.ov_eigvecs = loadOverlapEigenSystem(self.latt_info, file, use_fp32)
        self.ov_eigprec = eigprec

    def invert(
        self,
        b: MultiLatticeFermion,
        masses: Sequence[float],
        tol: float,
        maxiter: int,
        ov_ploy_prec: float,
        ov_use_fp32: float,
        ov_test: int,
        one_minus_half_d: int,
    ):
        return invertOverlap(
            b,
            self.gauge,
            self.kappa,
            masses,
            tol,
            maxiter,
            ov_ploy_prec,
            ov_use_fp32,
            ov_test,
            one_minus_half_d,
            self.hw_eigvals,
            self.hw_eigvecs,
            self.hw_eigprec,
            self.ov_eigvals,
            self.ov_eigvecs,
            self.ov_eigprec,
            False,
        )
