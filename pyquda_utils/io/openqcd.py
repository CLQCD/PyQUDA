from os import path
import struct
from typing import List

import numpy
from mpi4py import MPI

from .mpi_file import getSublatticeSize, getNeighbourRank, readMPIFile, writeMPIFile

Nd, Ns, Nc = 4, 4, 3


def cb2Gauge(latt_size: List[int], gauge: numpy.ndarray):
    Lx, Ly, Lz, Lt = latt_size
    gauge_cb2 = numpy.zeros_like(gauge).reshape(Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    gauge_cb2[:, 0, t, z, y, :] = gauge[:, t, z, y, 0::2]
                    gauge_cb2[:, 1, t, z, y, :] = gauge[:, t, z, y, 1::2]
                else:
                    gauge_cb2[:, 0, t, z, y, :] = gauge[:, t, z, y, 1::2]
                    gauge_cb2[:, 1, t, z, y, :] = gauge[:, t, z, y, 0::2]
    return gauge_cb2


def lexicoGauge(latt_size: List[int], gauge: numpy.ndarray):
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


def shiftGaugeOddForward(latt_size: List[int], grid_size: List[int], gauge: numpy.ndarray):
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


def shiftGaugeEvenBackward(latt_size: List[int], grid_size: List[int], gauge: numpy.ndarray):
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


def readGauge(filename: str, grid_size: List[int], lexico: bool = True):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        latt_size = struct.unpack("<iiii", f.read(16))[::-1]
        plaquette = struct.unpack("<d", f.read(8))[0] / Nc
        offset = f.tell()
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    dtype = "<c16"

    gauge_reorder = readMPIFile(filename, dtype, offset, (Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), (1, 2, 3, 0), grid_size)

    gauge = numpy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), dtype)
    for t in range(Lt):
        for y in range(Ly):
            for z in range(Lz):
                for x in range(Lx // 2):
                    x_ = 2 * x + (1 - (t + z + y) % 2)
                    z_ = z // 2
                    gauge[[3, 0, 1, 2], :, t, z, y, x, :, :] = gauge_reorder[t, x_, y, z_]

    gauge = shiftGaugeOddForward(latt_size, grid_size, gauge)
    if lexico:
        gauge = lexicoGauge([Lx, Ly, Lz, Lt], gauge).astype("<c16")
    return latt_size, plaquette, gauge


def writeGauge(
    filename: str,
    latt_size: List[int],
    grid_size: List[int],
    plaquette: float,
    gauge: numpy.ndarray,
    lexico: bool = True,
):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    dtype, offset = "<c16", None

    if lexico:
        gauge = cb2Gauge([Lx, Ly, Lz, Lt], gauge.astype(dtype))
    gauge = shiftGaugeEvenBackward(latt_size, grid_size, gauge)
    gauge_reorder = numpy.zeros((Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), dtype)
    for t in range(Lt):
        for y in range(Ly):
            for z in range(Lz):
                for x in range(Lx // 2):
                    x_ = 2 * x + (1 - (t + z + y) % 2)
                    z_ = z // 2
                    gauge_reorder[t, x_, y, z_] = gauge[[3, 0, 1, 2], :, t, z, y, x, :, :]

    gauge = gauge_reorder.astype(dtype)
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(filename, "wb") as f:
            f.write(struct.pack("<iiii", *latt_size[::-1]))
            f.write(struct.pack("<d", plaquette * Nc))
            offset = f.tell()
    offset = MPI.COMM_WORLD.bcast(offset)

    writeMPIFile(filename, dtype, offset, (Lt, Lx, Ly, Lz // 2, Nd, 2, Nc, Nc), (1, 2, 3, 0), grid_size, gauge)
