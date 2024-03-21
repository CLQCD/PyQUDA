import numpy

from ... import getMPIComm, getCoordFromRank
from ...field import Nd, Ns, Nc, LatticeInfo


def _reorderGauge(gauge_raw: numpy.ndarray, gauge_recv: numpy.ndarray, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lt, Lz, Ly, Lx = latt_info.size

    for rank in range(latt_info.mpi_size):
        gx, gy, gz, gt = getCoordFromRank(rank, [Gx, Gy, Gz, Gt])
        gauge_raw[
            :,
            gt * Lt : (gt + 1) * Lt,
            gz * Lz : (gz + 1) * Lz,
            gy * Ly : (gy + 1) * Ly,
            gx * Lx : (gx + 1) * Lx,
        ] = gauge_recv[rank]


def _reorder(data_raw: numpy.ndarray, data_recv: numpy.ndarray, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lt, Lz, Ly, Lx = latt_info.size
    for rank in range(latt_info.mpi_size):
        gx, gy, gz, gt = getCoordFromRank(rank, [Gx, Gy, Gz, Gt])
        data_raw[
            gt * Lt : (gt + 1) * Lt,
            gz * Lz : (gz + 1) * Lz,
            gy * Ly : (gy + 1) * Ly,
            gx * Lx : (gx + 1) * Lx,
        ] = data_recv[rank]


def gatherGaugeRaw(gauge_send: numpy.ndarray, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lt, Lz, Ly, Lx = latt_info.size
    dtype = gauge_send.dtype

    if latt_info.mpi_rank == 0:
        gauge_recv = numpy.zeros((Gt * Gz * Gy * Gx, Nd, Lt, Lz, Ly, Lx, Nc, Nc), dtype)
        getMPIComm().Gatherv(gauge_send, gauge_recv)
        gauge_raw = numpy.zeros((Nd, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc, Nc), dtype)
        _reorderGauge(gauge_raw, gauge_recv, latt_info)
    else:
        gauge_recv = None
        getMPIComm().Gatherv(gauge_send, gauge_recv)
        gauge_raw = None

    return gauge_raw


def gatherPropagatorRaw(propagator_send: numpy.ndarray, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lt, Lz, Ly, Lx = latt_info.size
    dtype = propagator_send.dtype

    if latt_info.mpi_rank == 0:
        propagator_recv = numpy.zeros((Gt * Gz * Gy * Gx, Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), dtype)
        getMPIComm().Gatherv(propagator_send, propagator_recv)
        propagator_raw = numpy.zeros((Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Ns, Nc, Nc), dtype)
        _reorder(propagator_raw, propagator_recv, latt_info)
    else:
        propagator_recv = None
        getMPIComm().Gatherv(propagator_send, propagator_recv)
        propagator_raw = None

    return propagator_raw


def gatherStaggeredPropagatorRaw(propagator_send: numpy.ndarray, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lt, Lz, Ly, Lx = latt_info.size
    dtype = propagator_send.dtype

    if latt_info.mpi_rank == 0:
        propagator_recv = numpy.zeros((Gt * Gz * Gy * Gx, Lt, Lz, Ly, Lx, Nc, Nc), dtype)
        getMPIComm().Gatherv(propagator_send, propagator_recv)
        propagator_raw = numpy.zeros((Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc, Nc), dtype)
        _reorder(propagator_raw, propagator_recv, latt_info)
    else:
        propagator_recv = None
        getMPIComm().Gatherv(propagator_send, propagator_recv)
        propagator_raw = None

    return propagator_raw


def gatherFermionRaw(fermion_send: numpy.ndarray, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lt, Lz, Ly, Lx = latt_info.size
    dtype = fermion_send.dtype

    if latt_info.mpi_rank == 0:
        fermion_recv = numpy.zeros((Gt * Gz * Gy * Gx, Lt, Lz, Ly, Lx, Ns, Nc), dtype)
        getMPIComm().Gatherv(fermion_send, fermion_recv)
        fermion_raw = numpy.zeros((Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc), dtype)
        _reorder(fermion_raw, fermion_recv, latt_info)
    else:
        fermion_recv = None
        getMPIComm().Gatherv(fermion_send, fermion_recv)
        fermion_raw = None

    return fermion_raw


def gatherStaggeredFermionRaw(fermion_send: numpy.ndarray, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lt, Lz, Ly, Lx = latt_info.size
    dtype = fermion_send.dtype

    if latt_info.mpi_rank == 0:
        fermion_recv = numpy.zeros((Gt * Gz * Gy * Gx, Lt, Lz, Ly, Lx, Nc), dtype)
        getMPIComm().Gatherv(fermion_send, fermion_recv)
        fermion_raw = numpy.zeros((Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc), dtype)
        _reorder(fermion_raw, fermion_recv, latt_info)
    else:
        fermion_recv = None
        getMPIComm().Gatherv(fermion_send, fermion_recv)
        fermion_raw = None

    return fermion_raw
