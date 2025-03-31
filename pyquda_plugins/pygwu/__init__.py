from os import path
from time import perf_counter
from typing import Sequence

import numpy as np

from pyquda_comm import initGrid, initDevice, getLogger, setGridMap, readMPIFile, readMPIFileInChunks
from pyquda_comm.field import evenodd, LatticeInfo, LatticeGauge, LatticeFermion, MultiLatticeFermion
from . import _pygwu as gwu

setGridMap("TZYX_FASTEST")


def init(latt_size: Sequence[int]):
    import atexit

    initGrid(None, latt_size)
    initDevice("numpy")
    gwu.gwu_init_machine(np.asarray(latt_size, "<i4"))
    atexit.register(gwu.gwu_shutdown_machine)


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


def negMultiFermionFromDiracPauli(dirac_pauli: MultiLatticeFermion):
    """Convert to the negative DeGrand-Rossi basis"""
    data = dirac_pauli.data / 2**0.5
    degrand_rossi = MultiLatticeFermion(dirac_pauli.latt_info, dirac_pauli.L5)
    degrand_rossi.data[:, :, :, :, :, :, 0, :] = -data[:, :, :, :, :, :, 1] + data[:, :, :, :, :, :, 3]
    degrand_rossi.data[:, :, :, :, :, :, 1, :] = data[:, :, :, :, :, :, 0] - data[:, :, :, :, :, :, 2]
    degrand_rossi.data[:, :, :, :, :, :, 2, :] = data[:, :, :, :, :, :, 1] + data[:, :, :, :, :, :, 3]
    degrand_rossi.data[:, :, :, :, :, :, 3, :] = -data[:, :, :, :, :, :, 0] - data[:, :, :, :, :, :, 2]
    return degrand_rossi


def negFermionFromDiracPauli(dirac_pauli: LatticeFermion):
    """Convert to the negative DeGrand-Rossi basis"""
    data = dirac_pauli.data / 2**0.5
    degrand_rossi = LatticeFermion(dirac_pauli.latt_info)
    degrand_rossi.data[:, :, :, :, :, 0, :] = -data[:, :, :, :, :, 1] + data[:, :, :, :, :, 3]
    degrand_rossi.data[:, :, :, :, :, 1, :] = data[:, :, :, :, :, 0] - data[:, :, :, :, :, 2]
    degrand_rossi.data[:, :, :, :, :, 2, :] = data[:, :, :, :, :, 1] + data[:, :, :, :, :, 3]
    degrand_rossi.data[:, :, :, :, :, 3, :] = -data[:, :, :, :, :, 0] - data[:, :, :, :, :, 2]
    return degrand_rossi


def readEigenValue(file: str):
    eigvals = []
    with open(file, "r") as f:
        for line in f.readlines():
            if line.startswith("EIGV"):
                tag, real, imag, res = line.strip().split()
                eigvals.append(float(real) + 1j * float(imag))
    return eigvals


def readEigenSystem(latt_info: LatticeInfo, eignum: int, file: str, use_fp32: bool):
    """Convert to the negative DeGrand-Rossi"""
    s = perf_counter()
    file = path.expanduser(path.expandvars(file))
    eigvals = readEigenValue(f"{file}.eigvals")
    assert eignum <= len(eigvals)
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
    eigvecs = negMultiFermionFromDiracPauli(eigvecs)
    getLogger().info(f"Read {eignum} eigen system in {perf_counter() - s:.3} secs")
    return eigvals[:eignum], eigvecs


def readEigenSystemInChunks(latt_info: LatticeInfo, eignum: int, file: str, use_fp32: bool):
    """Convert to the negative DeGrand-Rossi"""
    s = perf_counter()
    file = path.expanduser(path.expandvars(file))
    eigvals = readEigenValue(f"{file}.eigvals")
    assert eignum <= len(eigvals)
    Lx, Ly, Lz, Lt = latt_info.size
    Ns, Nc = latt_info.Ns, latt_info.Nc
    eigvecs = MultiLatticeFermion(latt_info, eignum)
    if use_fp32:
        for i, eigvecs_raw in readMPIFileInChunks(
            f"{file}.s", "<c8", 0, eignum, (Lt, Lz, Ly, Lx, Ns, Nc), (3, 2, 1, 0)
        ):
            eigvecs_raw = eigvecs_raw.astype("<c16")
            eigvecs[i] = negFermionFromDiracPauli(LatticeFermion(latt_info, evenodd(eigvecs_raw, [0, 1, 2, 3])))
    else:
        for i, eigvecs_raw in readMPIFileInChunks(file, ">f8", 0, eignum, (2, Ns, Nc, Lt, Lz, Ly, Lx), (6, 5, 4, 3)):
            eigvecs_raw = (
                eigvecs_raw.astype("<f8")
                .transpose(3, 4, 5, 6, 1, 2, 0)
                .reshape(Lt, Lz, Ly, Lx, Ns, Nc * 2)
                .copy()
                .view("<c16")
            )
            eigvecs[i] = negFermionFromDiracPauli(LatticeFermion(latt_info, evenodd(eigvecs_raw, [0, 1, 2, 3])))
    getLogger().info(f"Read {eignum} eigen system in {perf_counter() - s:.3} secs")
    return eigvals[:eignum], eigvecs


class Overlap:
    def __init__(self, latt_info: LatticeInfo):
        self.latt_info = latt_info

    def buildHWilson(self, gauge: LatticeGauge, kappa: float):
        gauge_in = gauge.copy()
        if self.latt_info.t_boundary == -1:
            gauge_in.setAntiPeriodicT()
        gwu.gwu_build_hw(gauge_in.data_ptr(0), kappa)

    def loadHWilsonEigen(self, eignum: int, eigprec: float, file: str, use_fp32: bool, chunk: bool = False):
        if chunk:
            eigvals, eigvecs = readEigenSystemInChunks(self.latt_info, eignum, file, use_fp32)
        else:
            eigvals, eigvecs = readEigenSystem(self.latt_info, eignum, file, use_fp32)
        gwu.gwu_load_hw_eigen(eignum, eigprec, np.asarray(eigvals, "<c16"), eigvecs.data_ptrs)

    def buildOverlap(self, ov_poly_prec: float, ov_use_fp32: int):
        gwu.gwu_build_ov(ov_poly_prec, ov_use_fp32)

    def loadOverlapEigen(self, eignum: int, eigprec: float, file: str, use_fp32: bool, chunk: bool = False):
        if chunk:
            eigvals, eigvecs = readEigenSystemInChunks(self.latt_info, eignum, file, use_fp32)
        else:
            eigvals, eigvecs = readEigenSystem(self.latt_info, eignum, file, use_fp32)
        gwu.gwu_load_ov_eigen(eignum, eigprec, np.asarray(eigvals, "<c16"), eigvecs.data_ptrs)

    def buildHWilsonEigen(
        self,
        eignum: int,
        eigprec: float,
        extra_krylov: int,
        maxiter: int,
        chebyshev_order: int,
        chebyshev_cut: float,
        iseed: int,
    ):
        gwu.gwu_build_hw_eigen(eignum, eigprec, extra_krylov, maxiter, chebyshev_order, chebyshev_cut, iseed)

    def invert(
        self,
        b: MultiLatticeFermion,
        masses: Sequence[float],
        tol: float,
        maxiter: int,
        one_minus_half_d: int,
        mode: int = 3,
        dirac_pauli: bool = False,
    ):
        latt_info = b.latt_info
        x = MultiLatticeFermion(latt_info, len(masses) * b.L5)
        if not dirac_pauli:
            b = multiFermionToDiracPauli(b)
        gwu.gwu_invert_overlap(
            x.data_ptrs, b.data_ptrs, len(masses), np.asarray(masses, "<f8"), tol, maxiter, one_minus_half_d, mode
        )
        if not dirac_pauli:
            x = multiFermionFromDiracPauli(x)
        return x
