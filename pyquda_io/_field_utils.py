from math import prod
from typing import List

from mpi4py import MPI
import numpy

from ._mpi_file import getMPIComm, getMPIRank, getSublatticeSize, getNeighbourRank

Nd, Ns, Nc = 4, 4, 3


def gaugeEvenOdd(sublatt_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = sublatt_size
    gauge_eo = numpy.zeros_like(gauge).reshape(Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    gauge_eo[:, 0, t, z, y, :] = gauge[:, t, z, y, 0::2]
                    gauge_eo[:, 1, t, z, y, :] = gauge[:, t, z, y, 1::2]
                else:
                    gauge_eo[:, 0, t, z, y, :] = gauge[:, t, z, y, 1::2]
                    gauge_eo[:, 1, t, z, y, :] = gauge[:, t, z, y, 0::2]
    return gauge_eo


def gaugeLexico(sublatt_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = sublatt_size
    gauge_lexico = numpy.empty_like(gauge).reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    gauge_lexico[:, t, z, y, 0::2] = gauge[:, 0, t, z, y, :]
                    gauge_lexico[:, t, z, y, 1::2] = gauge[:, 1, t, z, y, :]
                else:
                    gauge_lexico[:, t, z, y, 1::2] = gauge[:, 0, t, z, y, :]
                    gauge_lexico[:, t, z, y, 0::2] = gauge[:, 1, t, z, y, :]
    return gauge_lexico


def gaugePlaquette(latt_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    rank = getMPIRank()
    neighbour_rank = getNeighbourRank()
    extended = numpy.zeros_like(gauge, shape=(Nd, Lt + 1, Lz + 1, Ly + 1, Lx + 1, Nc, Nc))
    extended[:, :-1, :-1, :-1, :-1] = gauge
    if rank == neighbour_rank[0] and rank == neighbour_rank[4]:
        extended[:, :-1, :-1, :-1, -1] = gauge[:, :, :, :, 0]
    else:
        buf = gauge[:, :, :, :, 0].copy()
        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[4], source=neighbour_rank[0])
        extended[:, :-1, :-1, :-1, -1] = buf
    if rank == neighbour_rank[1] and rank == neighbour_rank[5]:
        extended[:, :-1, :-1, -1, :-1] = gauge[:, :, :, 0, :]
    else:
        buf = gauge[:, :, :, 0, :].copy()
        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[5], source=neighbour_rank[1])
        extended[:, :-1, :-1, -1, :-1] = buf
    if rank == neighbour_rank[2] and rank == neighbour_rank[6]:
        extended[:, :-1, -1, :-1, :-1] = gauge[:, :, 0, :, :]
    else:
        buf = gauge[:, :, 0, :, :].copy()
        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[6], source=neighbour_rank[2])
        extended[:, :-1, -1, :-1, :-1] = buf
    if rank == neighbour_rank[3] and rank == neighbour_rank[7]:
        extended[:, -1, :-1, :-1, :-1] = gauge[:, 0, :, :, :]
    else:
        buf = gauge[:, 0, :, :, :].copy()
        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[7], source=neighbour_rank[3])
        extended[:, -1, :-1, :-1, :-1] = buf

    plaq = numpy.empty((6), "<f8")
    plaq[0] = numpy.vdot(gauge[0] @ extended[1, :-1, :-1, :-1, 1:], gauge[1] @ extended[0, :-1, :-1, 1:, :-1]).real
    plaq[1] = numpy.vdot(gauge[0] @ extended[2, :-1, :-1, :-1, 1:], gauge[2] @ extended[0, :-1, 1:, :-1, :-1]).real
    plaq[2] = numpy.vdot(gauge[1] @ extended[2, :-1, :-1, 1:, :-1], gauge[2] @ extended[1, :-1, 1:, :-1, :-1]).real
    plaq[3] = numpy.vdot(gauge[0] @ extended[3, :-1, :-1, :-1, 1:], gauge[3] @ extended[0, 1:, :-1, :-1, :-1]).real
    plaq[4] = numpy.vdot(gauge[1] @ extended[3, :-1, :-1, 1:, :-1], gauge[3] @ extended[1, 1:, :-1, :-1, :-1]).real
    plaq[5] = numpy.vdot(gauge[2] @ extended[3, :-1, 1:, :-1, :-1], gauge[3] @ extended[2, 1:, :-1, :-1, :-1]).real
    plaq /= prod(latt_size) * Nc
    plaq = getMPIComm().allreduce(plaq, MPI.SUM)
    return numpy.array([plaq.mean(), plaq[:3].mean(), plaq[3:].mean()])


def gaugeOddShiftForward(latt_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    gauge_shift = numpy.empty_like(gauge)
    gauge_shift[:, 1] = gauge[:, 0]
    rank = getMPIRank()
    neighbour_rank = getNeighbourRank()
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                if (t + z + y) % 2 == 0:
                    gauge_shift[0, 0, t, z, y, :] = gauge[0, 1, t, z, y, :]
                else:
                    gauge_shift[0, 0, t, z, y, :-1] = gauge[0, 1, t, z, y, 1:]
                    if rank == neighbour_rank[0] and rank == neighbour_rank[4]:
                        gauge_shift[0, 0, t, z, y, -1] = gauge[0, 1, t, z, y, 0]
                    else:
                        buf = gauge[0, 1, t, z, y, 0].copy()
                        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[4], source=neighbour_rank[0])
                        gauge_shift[0, 0, t, z, y, -1] = buf
    gauge_shift[1, 0, :, :, :-1, :] = gauge[1, 1, :, :, 1:, :]
    if rank == neighbour_rank[1] and rank == neighbour_rank[5]:
        gauge_shift[1, 0, :, :, -1, :] = gauge[1, 1, :, :, 0, :]
    else:
        buf = gauge[1, 1, :, :, 0, :].copy()
        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[5], source=neighbour_rank[1])
        gauge_shift[1, 0, :, :, -1, :] = buf
    gauge_shift[2, 0, :, :-1, :, :] = gauge[2, 1, :, 1:, :, :]
    if rank == neighbour_rank[2] and rank == neighbour_rank[6]:
        gauge_shift[2, 0, :, -1, :, :] = gauge[2, 1, :, 0, :, :]
    else:
        buf = gauge[2, 1, :, 0, :, :].copy()
        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[6], source=neighbour_rank[2])
        gauge_shift[2, 0, :, -1, :, :] = buf
    gauge_shift[3, 0, :-1, :, :, :] = gauge[3, 1, 1:, :, :, :]
    if rank == neighbour_rank[3] and rank == neighbour_rank[7]:
        gauge_shift[3, 0, -1, :, :, :] = gauge[3, 1, 0, :, :, :]
    else:
        buf = gauge[3, 1, 0, :, :, :].copy()
        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[7], source=neighbour_rank[3])
        gauge_shift[3, 0, -1, :, :, :] = buf
    return gauge_shift


def gaugeEvenShiftBackward(latt_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    gauge_shift = numpy.empty_like(gauge)
    gauge_shift[:, 0] = gauge[:, 1]
    rank = getMPIRank()
    neighbour_rank = getNeighbourRank()
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                if (t + z + y) % 2 == 0:
                    gauge_shift[0, 1, t, z, y, :] = gauge[0, 0, t, z, y, :]
                else:
                    gauge_shift[0, 1, t, z, y, 1:] = gauge[0, 0, t, z, y, :-1]
                    if rank == neighbour_rank[0] and rank == neighbour_rank[4]:
                        gauge_shift[0, 1, t, z, y, 0] = gauge[0, 0, t, z, y, -1]
                    else:
                        buf = gauge[0, 0, t, z, y, -1].copy()
                        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[0], source=neighbour_rank[4])
                        gauge_shift[0, 1, t, z, y, 0] = buf
    gauge_shift[1, 1, :, :, 1:, :] = gauge[1, 0, :, :, :-1, :]
    if rank == neighbour_rank[1] and rank == neighbour_rank[5]:
        gauge_shift[1, 1, :, :, 0, :] = gauge[1, 0, :, :, -1, :]
    else:
        buf = gauge[1, 0, :, :, -1, :].copy()
        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[1], source=neighbour_rank[5])
        gauge_shift[1, 1, :, :, 0, :] = buf
    gauge_shift[2, 1, :, 1:, :, :] = gauge[2, 0, :, :-1, :, :]
    if rank == neighbour_rank[2] and rank == neighbour_rank[6]:
        gauge_shift[2, 1, :, 0, :, :] = gauge[2, 0, :, -1, :, :]
    else:
        buf = gauge[2, 0, :, -1, :, :].copy()
        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[2], source=neighbour_rank[6])
        gauge_shift[2, 1, :, 0, :, :] = buf
    gauge_shift[3, 1, 1:, :, :, :] = gauge[3, 0, :-1, :, :, :]
    if rank == neighbour_rank[3] and rank == neighbour_rank[7]:
        gauge_shift[3, 1, 0, :, :, :] = gauge[3, 0, -1, :, :, :]
    else:
        buf = gauge[3, 0, -1, :, :, :].copy()
        getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[3], source=neighbour_rank[7])
        gauge_shift[3, 1, 0, :, :, :] = buf
    return gauge_shift


def gaugeProject(gauge: numpy.ndarray):
    pass


def gaugeReunitarize(gauge: numpy.ndarray, reunitarize_sigma: float):
    gauge = numpy.ascontiguousarray(gauge.transpose(5, 6, 0, 1, 2, 3, 4))
    row0_abs = numpy.linalg.norm(gauge[0], axis=0)
    gauge[0] /= row0_abs
    row0_row1 = numpy.sum(gauge[0].conjugate() * gauge[1], axis=0)
    gauge[1] -= row0_row1 * gauge[0]
    row1_abs = numpy.linalg.norm(gauge[1], axis=0)
    gauge[1] /= row1_abs
    row2 = numpy.cross(gauge[0], gauge[1], axis=0).conjugate()
    if reunitarize_sigma > 0:
        sigma = numpy.sqrt(
            (1 - row0_abs) ** 2
            + numpy.abs(row0_row1) ** 2
            + (1 - row1_abs) ** 2
            + numpy.linalg.norm(row2 - gauge[2], axis=0) ** 2
        )
        failed = getMPIComm().allreduce(numpy.sum(sigma > reunitarize_sigma), MPI.SUM)
        assert failed == 0, f"Reunitarization failed {failed} times"
    gauge[2] = row2
    return gauge.transpose(2, 3, 4, 5, 6, 0, 1)


def gaugeReunitarizeReconstruct12(gauge: numpy.ndarray, reunitarize_sigma: float):
    """gauge shape (Nd, Lt, Lz, Ly, Lx, Nc - 1, Nc)"""
    gauge_ = gauge.transpose(5, 6, 0, 1, 2, 3, 4)
    gauge = numpy.empty((Nc, *gauge_.shape[1:]), "<c16")
    gauge[:2] = gauge_
    row0_abs = numpy.linalg.norm(gauge[0], axis=0)
    gauge[0] /= row0_abs
    row0_row1 = numpy.sum(gauge[0].conjugate() * gauge[1], axis=0)
    gauge[1] -= row0_row1 * gauge[0]
    row1_abs = numpy.linalg.norm(gauge[1], axis=0)
    gauge[1] /= row1_abs
    row2 = numpy.cross(gauge[0], gauge[1], axis=0).conjugate()
    if reunitarize_sigma > 0:
        sigma = numpy.sqrt((1 - row0_abs) ** 2 + numpy.abs(row0_row1) ** 2 + (1 - row1_abs) ** 2)
        failed = getMPIComm().allreduce(numpy.sum(sigma > reunitarize_sigma), MPI.SUM)
        assert failed == 0, f"Reunitarization failed {failed} times"
    gauge[2] = row2
    return gauge.transpose(2, 3, 4, 5, 6, 0, 1)


def gaugeReconstruct12(gauge: numpy.ndarray):
    """gauge shape (Nd, Lt, Lz, Ly, Lx, Nc - 1, Nc)"""
    gauge_ = gauge.transpose(5, 6, 0, 1, 2, 3, 4)
    gauge = numpy.empty((Nc, *gauge_.shape[1:]), "<c16")
    gauge[:2] = gauge_
    gauge[2] = numpy.cross(gauge[0], gauge[1], axis=0).conjugate()
    return gauge.transpose(2, 3, 4, 5, 6, 0, 1)


# matrices to convert gamma basis bewteen DeGrand-Rossi and Dirac-Pauli
# DP for Dirac-Pauli, DR for DeGrand-Rossi
# \psi(DP) = _DR_TO_DP \psi(DR)
# \psi(DR) = _DP_TO_DR \psi(DP)
_FROM_DIRAC_PAULI = numpy.array(
    [
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 1],
        [-1, 0, -1, 0],
    ],
    "<i4",
)
_TO_DIRAC_PAULI = numpy.array(
    [
        [0, -1, 0, -1],
        [1, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
    ],
    "<i4",
)


def propagatorFromDiracPauli2(propagator: numpy.ndarray):
    P = _FROM_DIRAC_PAULI
    Pinv = _TO_DIRAC_PAULI / 2

    return numpy.ascontiguousarray(numpy.einsum("ij,tzyxjkab,kl->tzyxilab", P, propagator.data, Pinv, optimize=True))


def propagatorToDiracPauli2(propagator: numpy.ndarray):
    P = _TO_DIRAC_PAULI
    Pinv = _FROM_DIRAC_PAULI / 2

    return numpy.ascontiguousarray(numpy.einsum("ij,tzyxjkab,kl->tzyxilab", P, propagator.data, Pinv, optimize=True))


def propagatorFromDiracPauli(dirac_pauli: numpy.ndarray):
    P = _FROM_DIRAC_PAULI
    if dirac_pauli.dtype.str == "<f8":  # Special case for KYU
        dirac_pauli = numpy.ascontiguousarray(dirac_pauli.transpose(4, 5, 0, 1, 2, 3, 6, 7) / 2).view("<c16")
    else:
        dirac_pauli = numpy.ascontiguousarray(dirac_pauli.transpose(4, 5, 0, 1, 2, 3, 6, 7) / 2)
    degrand_rossi = numpy.zeros_like(dirac_pauli)
    for i in range(4):
        for j in range(4):
            for i_ in range(4):
                for j_ in range(4):
                    if P[i, i_] * P[j, j_] == 1:
                        degrand_rossi[i, j] += dirac_pauli[i_, j_]
                    elif P[i, i_] * P[j, j_] == -1:
                        degrand_rossi[i, j] -= dirac_pauli[i_, j_]
    return degrand_rossi.transpose(2, 3, 4, 5, 0, 1, 6, 7)


def propagatorToDiracPauli(degrand_rossi: numpy.ndarray):
    P = _TO_DIRAC_PAULI
    degrand_rossi = numpy.ascontiguousarray(degrand_rossi.transpose(4, 5, 0, 1, 2, 3, 6, 7) / 2)
    dirac_pauli = numpy.zeros_like(degrand_rossi)
    for i in range(4):
        for j in range(4):
            for i_ in range(4):
                for j_ in range(4):
                    if P[i, i_] * P[j, j_] == 1:
                        dirac_pauli[i, j] += degrand_rossi[i_, j_]
                    elif P[i, i_] * P[j, j_] == -1:
                        dirac_pauli[i, j] -= degrand_rossi[i_, j_]
    return dirac_pauli.transpose(2, 3, 4, 5, 0, 1, 6, 7)


# def gaugeOddPlaqutteOpenQCD(latt_size: List[int], gauge: numpy.ndarray):
#     plaq = numpy.empty((6), "<f8")
#     plaq[0] = numpy.vdot(gauge[0, 1] @ gauge[1, 0], gauge[1, 1] @ gauge[0, 0]).real
#     plaq[1] = numpy.vdot(gauge[0, 1] @ gauge[2, 0], gauge[2, 1] @ gauge[0, 0]).real
#     plaq[2] = numpy.vdot(gauge[0, 1] @ gauge[3, 0], gauge[3, 1] @ gauge[0, 0]).real
#     plaq[3] = numpy.vdot(gauge[1, 1] @ gauge[2, 0], gauge[2, 1] @ gauge[1, 0]).real
#     plaq[4] = numpy.vdot(gauge[1, 1] @ gauge[3, 0], gauge[3, 1] @ gauge[1, 0]).real
#     plaq[5] = numpy.vdot(gauge[2, 1] @ gauge[3, 0], gauge[3, 1] @ gauge[2, 0]).real
#     plaq /= prod(latt_size) * Nc
#     plaq = getMPIComm().allreduce(plaq, MPI.SUM)
#     return numpy.array([plaq.mean(), plaq[:3].mean(), plaq[3:].mean()])


# def gaugeEvenPlaquette(latt_size: List[int], gauge: numpy.ndarray):
#     Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
#     link_munu = numpy.empty((6, *gauge.shape[2:]), gauge.dtype)
#     link_numu = numpy.empty((6, *gauge.shape[2:]), gauge.dtype)
#     rank = getMPIRank()
#     neighbour_rank = getNeighbourRank()
#     for t in range(Lt):
#         for z in range(Lz):
#             for y in range(Ly):
#                 if (t + z + y) % 2 == 0:
#                     link_munu[0, t, z, y, :] = gauge[0, 0, t, z, y, :] @ gauge[1, 1, t, z, y, :]
#                     link_munu[1, t, z, y, :] = gauge[0, 0, t, z, y, :] @ gauge[2, 1, t, z, y, :]
#                     link_munu[3, t, z, y, :] = gauge[0, 0, t, z, y, :] @ gauge[3, 1, t, z, y, :]
#                 else:
#                     link_munu[0, t, z, y, :-1] = gauge[0, 0, t, z, y, :-1] @ gauge[1, 1, t, z, y, 1:]
#                     link_munu[1, t, z, y, :-1] = gauge[0, 0, t, z, y, :-1] @ gauge[2, 1, t, z, y, 1:]
#                     link_munu[3, t, z, y, :-1] = gauge[0, 0, t, z, y, :-1] @ gauge[3, 1, t, z, y, 1:]
#                     if rank == neighbour_rank[0] and rank == neighbour_rank[4]:
#                         link_munu[0, t, z, y, -1] = gauge[0, 0, t, z, y, -1] @ gauge[1, 1, t, z, y, 0]
#                         link_munu[1, t, z, y, -1] = gauge[0, 0, t, z, y, -1] @ gauge[2, 1, t, z, y, 0]
#                         link_munu[3, t, z, y, -1] = gauge[0, 0, t, z, y, -1] @ gauge[3, 1, t, z, y, 0]
#                     else:
#                         buf = gauge[:, 1, t, z, y, 0].copy()
#                         getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[4], source=neighbour_rank[0])
#                         link_munu[0, t, z, y, -1] = gauge[0, 0, t, z, y, -1] @ buf[1]
#                         link_munu[1, t, z, y, -1] = gauge[0, 0, t, z, y, -1] @ buf[2]
#                         link_munu[3, t, z, y, -1] = gauge[0, 0, t, z, y, -1] @ buf[3]
#     link_numu[0, :, :, :-1, :] = gauge[1, 0, :, :, :-1, :] @ gauge[0, 1, :, :, 1:, :]
#     link_munu[2, :, :, :-1, :] = gauge[1, 0, :, :, :-1, :] @ gauge[2, 1, :, :, 1:, :]
#     link_munu[4, :, :, :-1, :] = gauge[1, 0, :, :, :-1, :] @ gauge[3, 1, :, :, 1:, :]
#     if rank == neighbour_rank[1] and rank == neighbour_rank[5]:
#         link_numu[0, :, :, -1, :] = gauge[1, 0, :, :, -1, :] @ gauge[0, 1, :, :, 0, :]
#         link_munu[2, :, :, -1, :] = gauge[1, 0, :, :, -1, :] @ gauge[2, 1, :, :, 0, :]
#         link_munu[4, :, :, -1, :] = gauge[1, 0, :, :, -1, :] @ gauge[3, 1, :, :, 0, :]
#     else:
#         buf = gauge[:, 1, :, :, 0, :].copy()
#         getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[5], source=neighbour_rank[1])
#         link_numu[0, :, :, -1, :] = gauge[1, 0, :, :, -1, :] @ buf[0]
#         link_munu[2, :, :, -1, :] = gauge[1, 0, :, :, -1, :] @ buf[2]
#         link_munu[4, :, :, -1, :] = gauge[1, 0, :, :, -1, :] @ buf[3]
#     link_numu[1, :, :-1, :, :] = gauge[2, 0, :, :-1, :, :] @ gauge[0, 1, :, 1:, :, :]
#     link_numu[2, :, :-1, :, :] = gauge[2, 0, :, :-1, :, :] @ gauge[1, 1, :, 1:, :, :]
#     link_munu[5, :, :-1, :, :] = gauge[2, 0, :, :-1, :, :] @ gauge[3, 1, :, 1:, :, :]
#     if rank == neighbour_rank[2] and rank == neighbour_rank[6]:
#         link_numu[1, :, -1, :, :] = gauge[2, 0, :, -1, :, :] @ gauge[0, 1, :, 0, :, :]
#         link_numu[2, :, -1, :, :] = gauge[2, 0, :, -1, :, :] @ gauge[1, 1, :, 0, :, :]
#         link_munu[5, :, -1, :, :] = gauge[2, 0, :, -1, :, :] @ gauge[3, 1, :, 0, :, :]
#     else:
#         buf = gauge[:, 1, :, 0, :, :].copy()
#         getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[6], source=neighbour_rank[2])
#         link_numu[1, :, -1, :, :] = gauge[2, 0, :, -1, :, :] @ buf[0]
#         link_numu[2, :, -1, :, :] = gauge[2, 0, :, -1, :, :] @ buf[1]
#         link_munu[5, :, -1, :, :] = gauge[2, 0, :, -1, :, :] @ buf[3]
#     link_numu[3, :-1, :, :, :] = gauge[3, 0, :-1, :, :, :] @ gauge[0, 1, 1:, :, :, :]
#     link_numu[4, :-1, :, :, :] = gauge[3, 0, :-1, :, :, :] @ gauge[1, 1, 1:, :, :, :]
#     link_numu[5, :-1, :, :, :] = gauge[3, 0, :-1, :, :, :] @ gauge[2, 1, 1:, :, :, :]
#     if rank == neighbour_rank[3] and rank == neighbour_rank[7]:
#         link_numu[3, -1, :, :, :] = gauge[3, 0, -1, :, :, :] @ gauge[0, 1, 0, :, :, :]
#         link_numu[4, -1, :, :, :] = gauge[3, 0, -1, :, :, :] @ gauge[1, 1, 0, :, :, :]
#         link_numu[5, -1, :, :, :] = gauge[3, 0, -1, :, :, :] @ gauge[2, 1, 0, :, :, :]
#     else:
#         buf = gauge[3, 1, 0, :, :, :].copy()
#         getMPIComm().Sendrecv_replace(buf, dest=neighbour_rank[7], source=neighbour_rank[3])
#         link_numu[3, -1, :, :, :] = gauge[3, 0, -1, :, :, :] @ buf[0]
#         link_numu[4, -1, :, :, :] = gauge[3, 0, -1, :, :, :] @ buf[1]
#         link_numu[5, -1, :, :, :] = gauge[3, 0, -1, :, :, :] @ buf[2]

#     plaq = numpy.empty((6), "<f8")
#     plaq[0] = numpy.vdot(link_munu[0], link_numu[0]).real
#     plaq[1] = numpy.vdot(link_munu[1], link_numu[1]).real
#     plaq[2] = numpy.vdot(link_munu[2], link_numu[2]).real
#     plaq[3] = numpy.vdot(link_munu[3], link_numu[3]).real
#     plaq[4] = numpy.vdot(link_munu[4], link_numu[4]).real
#     plaq[5] = numpy.vdot(link_munu[5], link_numu[5]).real
#     plaq /= prod(latt_size) * Nc
#     plaq = getMPIComm().allreduce(plaq, MPI.SUM)
#     return numpy.array([plaq.mean(), plaq[:3].mean(), plaq[3:].mean()])
