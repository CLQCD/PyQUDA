from typing import List

from mpi4py import MPI
import numpy

from .mpi_file import getSublatticeSize, getNeighbourRank

Nd, Nc = 4, 3


def gaugeEvenOdd(latt_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = latt_size
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


def gaugeLexico(latt_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = latt_size
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


def gaugeLexicoPlaquette(latt_size: List[int], grid_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    rank = MPI.COMM_WORLD.Get_rank()
    neighbour_rank = getNeighbourRank(grid_size)
    extended = numpy.zeros_like(gauge, shape=(Nd, Lt + 1, Lz + 1, Ly + 1, Lx + 1, Nc, Nc))
    extended[:, :-1, :-1, :-1, :-1] = gauge
    if rank == neighbour_rank[0] and rank == neighbour_rank[4]:
        extended[:, :-1, :-1, :-1, -1] = gauge[:, :, :, :, 0]
    else:
        buf = gauge[:, :, :, :, 0].copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[4], source=neighbour_rank[0])
        extended[:, :-1, :-1, :-1, -1] = buf
    if rank == neighbour_rank[1] and rank == neighbour_rank[5]:
        extended[:, :-1, :-1, -1, :-1] = gauge[:, :, :, 0, :]
    else:
        buf = gauge[:, :, :, 0, :].copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[5], source=neighbour_rank[1])
        extended[:, :-1, :-1, -1, :-1] = buf
    if rank == neighbour_rank[2] and rank == neighbour_rank[6]:
        extended[:, :-1, -1, :-1, :-1] = gauge[:, :, 0, :, :]
    else:
        buf = gauge[:, :, 0, :, :].copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[6], source=neighbour_rank[2])
        extended[:, :-1, -1, :-1, :-1] = buf
    if rank == neighbour_rank[3] and rank == neighbour_rank[7]:
        extended[:, -1, :-1, :-1, :-1] = gauge[:, 0, :, :, :]
    else:
        buf = gauge[:, 0, :, :, :].copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[7], source=neighbour_rank[3])
        extended[:, -1, :-1, :-1, :-1] = buf

    plaq = numpy.zeros((6))
    plaq[0] = numpy.vdot(
        numpy.linalg.matmul(gauge[0], extended[1, :-1, :-1, :-1, 1:]),
        numpy.linalg.matmul(gauge[1], extended[0, :-1, :-1, 1:, :-1]),
    ).real
    plaq[1] = numpy.vdot(
        numpy.linalg.matmul(gauge[0], extended[2, :-1, :-1, :-1, 1:]),
        numpy.linalg.matmul(gauge[2], extended[0, :-1, 1:, :-1, :-1]),
    ).real
    plaq[2] = numpy.vdot(
        numpy.linalg.matmul(gauge[1], extended[2, :-1, :-1, 1:, :-1]),
        numpy.linalg.matmul(gauge[2], extended[1, :-1, 1:, :-1, :-1]),
    ).real
    plaq[3] = numpy.vdot(
        numpy.linalg.matmul(gauge[0], extended[3, :-1, :-1, :-1, 1:]),
        numpy.linalg.matmul(gauge[3], extended[0, 1:, :-1, :-1, :-1]),
    ).real
    plaq[4] = numpy.vdot(
        numpy.linalg.matmul(gauge[1], extended[3, :-1, :-1, 1:, :-1]),
        numpy.linalg.matmul(gauge[3], extended[1, 1:, :-1, :-1, :-1]),
    ).real
    plaq[5] = numpy.vdot(
        numpy.linalg.matmul(gauge[2], extended[3, :-1, 1:, :-1, :-1]),
        numpy.linalg.matmul(gauge[3], extended[2, 1:, :-1, :-1, :-1]),
    ).real

    plaq /= int(numpy.prod(latt_size)) * Nc
    plaq = MPI.COMM_WORLD.allreduce(plaq, MPI.SUM)
    return [plaq.mean().item(), plaq[:3].mean().item(), plaq[3:].mean().item()]


def gaugeOddShiftForward(latt_size: List[int], grid_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    gauge_shift = numpy.empty_like(gauge)
    gauge_shift[:, 1] = gauge[:, 0]
    rank = MPI.COMM_WORLD.Get_rank()
    neighbour_rank = getNeighbourRank(grid_size)
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
                        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[4], source=neighbour_rank[0])
                        gauge_shift[0, 0, t, z, y, -1] = buf
    gauge_shift[1, 0, :, :, :-1, :] = gauge[1, 1, :, :, 1:, :]
    if rank == neighbour_rank[1] and rank == neighbour_rank[5]:
        gauge_shift[1, 0, :, :, -1, :] = gauge[1, 1, :, :, 0, :]
    else:
        buf = gauge[1, 1, :, :, 0, :].copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[5], source=neighbour_rank[1])
        gauge_shift[1, 0, :, :, -1, :] = buf
    gauge_shift[2, 0, :, :-1, :, :] = gauge[2, 1, :, 1:, :, :]
    if rank == neighbour_rank[2] and rank == neighbour_rank[6]:
        gauge_shift[2, 0, :, -1, :, :] = gauge[2, 1, :, 0, :, :]
    else:
        buf = gauge[2, 1, :, 0, :, :].copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[6], source=neighbour_rank[2])
        gauge_shift[2, 0, :, -1, :, :] = buf
    gauge_shift[3, 0, :-1, :, :, :] = gauge[3, 1, 1:, :, :, :]
    if rank == neighbour_rank[3] and rank == neighbour_rank[7]:
        gauge_shift[3, 0, -1, :, :, :] = gauge[3, 1, 0, :, :, :]
    else:
        buf = gauge[3, 1, 0, :, :, :].copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[7], source=neighbour_rank[3])
        gauge_shift[3, 0, -1, :, :, :] = buf
    return gauge_shift


def gaugeEvenShiftBackward(latt_size: List[int], grid_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    gauge_shift = numpy.empty_like(gauge)
    gauge_shift[:, 0] = gauge[:, 1]
    rank = MPI.COMM_WORLD.Get_rank()
    neighbour_rank = getNeighbourRank(grid_size)
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
                        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[0], source=neighbour_rank[4])
                        gauge_shift[0, 1, t, z, y, 0] = buf
    gauge_shift[1, 1, :, :, 1:, :] = gauge[1, 0, :, :, :-1, :]
    if rank == neighbour_rank[1] and rank == neighbour_rank[5]:
        gauge_shift[1, 1, :, :, 0, :] = gauge[1, 0, :, :, -1, :]
    else:
        buf = gauge[1, 0, :, :, -1, :].copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[1], source=neighbour_rank[5])
        gauge_shift[1, 1, :, :, 0, :] = buf
    gauge_shift[2, 1, :, 1:, :, :] = gauge[2, 0, :, :-1, :, :]
    if rank == neighbour_rank[2] and rank == neighbour_rank[6]:
        gauge_shift[2, 1, :, 0, :, :] = gauge[2, 0, :, -1, :, :]
    else:
        buf = gauge[2, 0, :, -1, :, :].copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[2], source=neighbour_rank[6])
        gauge_shift[2, 1, :, 0, :, :] = buf
    gauge_shift[3, 1, 1:, :, :, :] = gauge[3, 0, :-1, :, :, :]
    if rank == neighbour_rank[3] and rank == neighbour_rank[7]:
        gauge_shift[3, 1, 0, :, :, :] = gauge[3, 0, -1, :, :, :]
    else:
        buf = gauge[3, 0, -1, :, :, :].copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=neighbour_rank[3], source=neighbour_rank[7])
        gauge_shift[3, 1, 0, :, :, :] = buf
    return gauge_shift
