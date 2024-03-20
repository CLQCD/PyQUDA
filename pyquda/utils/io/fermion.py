from os import path

import numpy

from ...field import Ns, Nc, LatticeInfo, LatticeFermion, cb2


def gatherFermionRaw(fermion_send: numpy.ndarray, latt_info: LatticeInfo):
    from ... import getMPIComm, getCoordFromRank

    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lt, Lz, Ly, Lx = latt_info.size
    dtype = fermion_send.dtype

    if latt_info.mpi_rank == 0:
        fermion_recv = numpy.zeros((Gt * Gz * Gy * Gx, Lt, Lz, Ly, Lx, Ns, Nc), dtype)
        getMPIComm().Gatherv(fermion_send, fermion_recv)

        fermion_raw = numpy.zeros((Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc), dtype)
        for rank in range(latt_info.mpi_size):
            gx, gy, gz, gt = getCoordFromRank(rank, [Gx, Gy, Gz, Gt])
            fermion_raw[
                gt * Lt : (gt + 1) * Lt,
                gz * Lz : (gz + 1) * Lz,
                gy * Ly : (gy + 1) * Ly,
                gx * Lx : (gx + 1) * Lx,
            ] = fermion_recv[rank]
    else:
        fermion_recv = None
        getMPIComm().Gatherv(fermion_send, fermion_recv)

        fermion_raw = None

    return fermion_raw


def fromKYUBuffer(buffer: bytes, dtype: str, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    fermion_raw = (
        numpy.frombuffer(buffer, dtype)
        .reshape(2, Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx)[
            :,
            :,
            :,
            gt * Lt : (gt + 1) * Lt,
            gz * Lz : (gz + 1) * Lz,
            gy * Ly : (gy + 1) * Ly,
            gx * Lx : (gx + 1) * Lx,
        ]
        .astype("<f8")
        .transpose(3, 4, 5, 6, 1, 2, 0)
        .reshape(Lt, Lz, Ly, Lx, Ns, Nc * 2)
        .view("<c16")
    )

    return fermion_raw


def toKYUBuffer(fermion_lexico: numpy.ndarray, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lx, Ly, Lz, Lt = latt_info.size

    fermion_raw = gatherFermionRaw(fermion_lexico, latt_info)
    if latt_info.mpi_rank == 0:
        buffer = (
            fermion_raw.view("<f8")
            .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc, 2)
            .transpose(6, 4, 5, 0, 1, 2, 3)
            .astype(">f8")
            .tobytes()
        )
    else:
        buffer = None

    return buffer


def readKYU(filename: str, latt_info: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        # kyu_binary_data = f.read(2 * Ns * Nc * Lt * Lz * Ly * Lx * 8)
        kyu_binary_data = f.read()
    fermion_raw = fromKYUBuffer(kyu_binary_data, ">f8", latt_info)

    return LatticeFermion(latt_info, cb2(fermion_raw, [0, 1, 2, 3]))


def writeKYU(filename: str, fermion: LatticeFermion):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = fermion.latt_info
    kyu_binary_data = toKYUBuffer(fermion.lexico(), latt_info)
    if latt_info.mpi_rank == 0:
        with open(filename, "wb") as f:
            f.write(kyu_binary_data)
