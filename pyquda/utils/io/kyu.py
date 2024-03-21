from os import path

import numpy

from ...field import Ns, Nc, Nd, LatticeInfo, LatticeGauge, LatticeFermion, cb2


def fromGaugeBuffer(buffer: bytes, dtype: str, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    gauge_raw = (
        numpy.frombuffer(buffer, dtype)
        .reshape(Nd, Nc, Nc, 2, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx)[
            :,
            :,
            :,
            :,
            gt * Lt : (gt + 1) * Lt,
            gz * Lz : (gz + 1) * Lz,
            gy * Ly : (gy + 1) * Ly,
            gx * Lx : (gx + 1) * Lx,
        ]
        .astype("<f8")
        .transpose(0, 4, 5, 6, 7, 2, 1, 3)
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2)
        .view("<c16")
    )

    return gauge_raw


def toGaugeBuffer(gauge_lexico: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from .gather_raw import gatherGaugeRaw

    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lx, Ly, Lz, Lt = latt_info.size

    gauge_raw = gatherGaugeRaw(gauge_lexico, latt_info)
    if latt_info.mpi_rank == 0:
        buffer = (
            gauge_raw.view("<f8")
            .reshape(Nd, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc, Nc, 2)
            .transpose(0, 6, 5, 7, 1, 2, 3, 4)
            .astype(dtype)
            .tobytes()
        )
    else:
        buffer = None

    return buffer


def readGauge(filename: str, latt_info: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        # kyu_binary_data = f.read(Nd * Nc * Nc * 2 * Lt * Lz * Ly * Lx * 8)
        kyu_binary_data = f.read()
    gauge_raw = fromGaugeBuffer(kyu_binary_data, ">f8", latt_info)

    return LatticeGauge(latt_info, cb2(gauge_raw, [1, 2, 3, 4]))


def writeGauge(filename: str, gauge: LatticeGauge):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = gauge.latt_info
    kyu_binary_data = toGaugeBuffer(gauge.lexico(), ">f8", latt_info)
    if latt_info.mpi_rank == 0:
        with open(filename, "wb") as f:
            f.write(kyu_binary_data)


def fromPropagatorBuffer(buffer: bytes, dtype: str, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    fermion_raw = (
        numpy.frombuffer(buffer, dtype)
        .reshape(Ns, Nc, 2, Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx)[
            :,
            :,
            :,
            :,
            :,
            gt * Lt : (gt + 1) * Lt,
            gz * Lz : (gz + 1) * Lz,
            gy * Ly : (gy + 1) * Ly,
            gx * Lx : (gx + 1) * Lx,
        ]
        .astype("<f8")
        .transpose(5, 6, 7, 8, 3, 0, 4, 1, 2)
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc * 2)
        .view("<c16")
    )

    return fermion_raw


def toPropagatorBuffer(fermion_lexico: numpy.ndarray, latt_info: LatticeInfo):
    from .gather_raw import gatherPropagatorRaw

    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lx, Ly, Lz, Lt = latt_info.size

    fermion_raw = gatherPropagatorRaw(fermion_lexico, latt_info)
    if latt_info.mpi_rank == 0:
        buffer = (
            fermion_raw.view("<f8")
            .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Ns, Nc, Nc, 2)
            .transpose(5, 7, 8, 4, 6, 0, 1, 2, 3)
            .astype(">f8")
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
    fermion_raw = fromPropagatorBuffer(kyu_binary_data, ">f8", latt_info)

    return LatticeFermion(latt_info, cb2(fermion_raw, [0, 1, 2, 3]))


def writePropagator(filename: str, fermion: LatticeFermion):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = fermion.latt_info
    kyu_binary_data = toPropagatorBuffer(fermion.lexico(), latt_info)
    if latt_info.mpi_rank == 0:
        with open(filename, "wb") as f:
            f.write(kyu_binary_data)
