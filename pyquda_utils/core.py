from math import prod
from typing import List, Literal, Union

import numpy

from pyquda_comm import getCoordFromRank, getRankFromCoord
from pyquda import (
    initGrid,
    initDevice,
    initQUDA,
    init,
    getMPIComm,
    getMPISize,
    getMPIRank,
    getGridSize,
    getGridCoord,
    getGridMap,
    setDefaultLattice,
    getDefaultLattice,
    getCUDABackend,
    getCUDADevice,
    getCUDAComputeCapability,
    getLogger,
    setLoggerLevel,
    dirac as fermion,
)
from pyquda.field import (  # noqa: F401
    Ns,
    Nc,
    Nd,
    X,
    Y,
    Z,
    T,
    LatticeInfo,
    LatticeInt,
    LatticeReal,
    LatticeComplex,
    LatticeLink,
    LatticeGauge,
    LatticeMom,
    LatticeFermion,
    MultiLatticeFermion,
    LatticeStaggeredFermion,
    MultiLatticeStaggeredFermion,
    LatticePropagator,
    LatticeStaggeredPropagator,
    lexico,
    evenodd,
)
from pyquda.dirac.abstract import Multigrid, FermionDirac, StaggeredFermionDirac

from . import source
from .deprecated import smear, smear4, invert12, getDslash, getStaggeredDslash, cb2

LaplaceLatticeInfo = LatticeInfo


def invert(
    dirac: FermionDirac,
    source_type: Literal["point", "wall", "volume", "momentum", "colorvector"],
    t_srce: Union[List[int], int, None],
    source_phase=None,
    restart: int = 0,
):
    latt_info = dirac.latt_info

    propag = LatticePropagator(latt_info)
    for spin in range(Ns):
        for color in range(Nc):
            b = source.source(latt_info, source_type, t_srce, spin, color, source_phase)
            x = dirac.invertRestart(b, restart)
            propag.setFermion(x, spin, color)

    return propag


def invertPropagator(
    dirac: FermionDirac,
    source_propag: LatticePropagator,
    restart: int = 0,
):
    latt_info = dirac.latt_info

    propag = LatticePropagator(latt_info)
    for spin in range(Ns):
        for color in range(Nc):
            b = source_propag.getFermion(spin, color)
            x = dirac.invertRestart(b, restart)
            propag.setFermion(x, spin, color)

    return propag


def invertStaggered(
    dirac: StaggeredFermionDirac,
    source_type: Literal["point", "wall", "volume", "momentum", "colorvector"],
    t_srce: Union[List[int], int, None],
    source_phase=None,
    restart: int = 0,
):
    latt_info = dirac.latt_info

    propag = LatticeStaggeredPropagator(latt_info)
    for color in range(Nc):
        b = source.source(latt_info, source_type, t_srce, None, color, source_phase)
        x = dirac.invertRestart(b, restart)
        propag.setFermion(x, color)

    return propag


def invertStaggeredPropagator(
    dirac: StaggeredFermionDirac,
    source_propag: LatticeStaggeredPropagator,
    restart: int = 0,
):
    latt_info = dirac.latt_info

    propag = LatticeStaggeredPropagator(latt_info)
    for color in range(Nc):
        b = source_propag.getFermion(color)
        x = dirac.invertRestart(b, restart)
        propag.setFermion(x, color)

    return propag


def gatherLattice2(
    data: numpy.ndarray, tzyx: List[int], reduce_op: Literal["sum", "mean", "prod", "max", "min"] = "sum", root: int = 0
):
    sendobj = numpy.ascontiguousarray(data)
    recvobj = getMPIComm().gather(sendobj, root)

    if getMPIRank() == root:
        gather_axis = [d for d in range(4)]
        reduce_axis = [d for d in range(4) if tzyx[::-1][d] < 0]
        grid_size = numpy.array(getGridSize())[gather_axis]
        send_latt = numpy.array([data.shape[i] if i >= 0 else 1 for i in tzyx[::-1]])[gather_axis]
        recv_latt = [G * L for G, L in zip(grid_size, send_latt)]

        keep = tuple([i for i in tzyx if i >= 0])
        keep = (0, -1) if keep == () else keep
        prefix = data.shape[: keep[0]]
        suffix = data.shape[keep[-1] + 1 :]
        prefix_slice = [slice(None) for _ in range(len(prefix))]
        suffix_slice = [slice(None) for _ in range(len(suffix))]

        data_all = numpy.zeros((*prefix, *recv_latt[::-1], *suffix), data.dtype)
        for rank in range(getMPISize()):
            grid_coord = numpy.array(getCoordFromRank(rank, getGridSize()))[gather_axis]
            recv_slice = [slice(g * L, (g + 1) * L) for g, L in zip(grid_coord, send_latt)]
            all_slice = (*prefix_slice, *recv_slice[::-1], *suffix_slice)
            data_all[all_slice] = recvobj[rank].reshape(*prefix, *send_latt[::-1], *suffix)

        reduce_axis = tuple([len(prefix) + 3 - axis for axis in reduce_axis])
        if reduce_op.lower() == "sum":
            recvobj = numpy.sum(data_all, reduce_axis)
        elif reduce_op.lower() == "mean":
            recvobj = numpy.mean(data_all, reduce_axis)
        elif reduce_op.lower() == "prod":
            recvobj = numpy.prod(data_all, reduce_axis)
        elif reduce_op.lower() == "max":
            recvobj = numpy.amax(data_all, reduce_axis)
        elif reduce_op.lower() == "min":
            recvobj = numpy.amin(data_all, reduce_axis)
        else:
            getLogger().critical(
                f"gatherLattice doesn't support reduce operator reduce_op={reduce_op}", NotImplementedError
            )

    return recvobj


def scatterLattice(data: numpy.ndarray, tzyx: List[int], root: int = 0):
    if getMPIRank() == root:
        scatter_axis = [d for d in range(4) if tzyx[::-1][d] >= 0]
        grid_size = numpy.array(getGridSize())[scatter_axis]
        send_latt = numpy.array([data.shape[i] if i >= 0 else 1 for i in tzyx[::-1]])[scatter_axis]
        recv_latt = [L // G for G, L in zip(grid_size, send_latt)]

        keep = tuple([i for i in tzyx if i >= 0])
        keep = (0, -1) if keep == () else keep
        prefix = data.shape[: keep[0]]
        suffix = data.shape[keep[-1] + 1 :]
        prefix_slice = [slice(None) for _ in range(len(prefix))]
        suffix_slice = [slice(None) for _ in range(len(suffix))]

        sendobj = []
        for rank in range(getMPISize()):
            grid_coord = numpy.array(getCoordFromRank(rank, getGridSize()))[scatter_axis]
            send_slice = [slice(g * L, (g + 1) * L) for g, L in zip(grid_coord, recv_latt)]
            all_slice = (*prefix_slice, *send_slice[::-1], *suffix_slice)
            sendobj.append(numpy.ascontiguousarray(data[all_slice]))
    else:
        sendobj = None

    recvobj = getMPIComm().scatter(sendobj, root)

    return recvobj


def gatherScatterLattice(
    data: numpy.ndarray, tzyx: List[int], reduce_op: Literal["sum", "mean", "prod", "max", "min"] = "sum", root: int = 0
):
    data = gatherLattice2(data, tzyx, reduce_op, root)
    data = scatterLattice(data, tzyx, root)
    return data


def gatherLattice(data: numpy.ndarray, axes: List[int], reduce_op: Literal["sum", "mean"] = "sum", root: int = 0):
    """
    MPI gather or reduce data from all MPI subgrid onto the root process.

    Args:
    - data: numpy.ndarray
        The local data array to be gathered.
    - axes: List[int]
        A list of length 4 specifying the axes along with the data gathered.
        Axes order should be (t z y x).
        Use axis >= 0 for gather lattice data along this axis direction.
            Warning: In this case, the length of the time / space axes
                times grid_size should match the global lattice shape.
        Use axis = -1 for the dimensions to which reduce_op mode should be applied.
    - reduce_op: Literal["sum", "mean"], optional
        The reduction operation to be applied after gathering the datai when its axis == -1. Default is "sum".
    - root: int, optional
        The rank of the root process that will receive the gathered data. Default is 0.

    Returns:
    - numpy.ndarray
        The gathered and reduced data array on the root process.

    Raises:
    - NotImplementedError
        If the specified reduce operation is not supported.

    Note:
    - This function assumes that MPI environment has been initialized before its invocation.
    """
    Gx, Gy, Gz, Gt = getGridSize()
    Lt, Lz, Ly, Lx = [data.shape[axis] if axis >= 0 else 1 for axis in axes]
    keep = tuple([axis for axis in axes if axis >= 0])
    keep = (0, -1) if keep == () else keep
    reduce_axis = tuple([keep[0] + d for d in range(4) if axes[d] == -1])
    prefix = data.shape[: keep[0]]
    suffix = data.shape[keep[-1] + 1 :]
    prefix_size = prod(prefix)
    suffix_size = prod(suffix)

    if getMPIRank() == root:
        sendbuf = numpy.ascontiguousarray(data.reshape(-1))
        recvbuf = numpy.zeros((getMPISize(), data.size), data.dtype)
        getMPIComm().Gatherv(sendbuf, recvbuf, root)

        data = numpy.zeros_like(recvbuf).reshape(prefix_size, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, suffix_size)
        for rank in range(getMPISize()):
            gx, gy, gz, gt = getCoordFromRank(rank, getGridSize())
            data[
                :,
                gt * Lt : (gt + 1) * Lt,
                gz * Lz : (gz + 1) * Lz,
                gy * Ly : (gy + 1) * Ly,
                gx * Lx : (gx + 1) * Lx,
                :,
            ] = recvbuf[rank].reshape(prefix_size, Lt, Lz, Ly, Lx, suffix_size)
        data = data.reshape(*prefix, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, *suffix)

        if reduce_op.lower() == "sum":
            return data.sum(reduce_axis)
        elif reduce_op.lower() == "mean":
            return data.mean(reduce_axis)
        else:
            getLogger().critical(
                f"core.gather doesn't support reduce operator reduce_op={reduce_op}", NotImplementedError
            )
    else:
        sendbuf = numpy.ascontiguousarray(data.reshape(-1))
        recvbuf = None
        getMPIComm().Gatherv(sendbuf, recvbuf, root)
        return None


def getDirac(
    latt_info: LatticeInfo,
    mass: float,
    tol: float,
    maxiter: int,
    xi_0: float = 1.0,
    clover_coeff_t: float = 0.0,
    clover_coeff_r: float = 1.0,
    multigrid: Union[List[List[int]], Multigrid] = None,
):
    xi = latt_info.anisotropy
    if xi != 1.0:
        clover_csw = xi_0 * clover_coeff_t**2 / clover_coeff_r
        clover_xi = (xi_0 * clover_coeff_t / clover_coeff_r) ** 0.5
    else:
        clover_csw = clover_coeff_t
        clover_xi = 1.0
    if not multigrid:
        multigrid = None
    else:
        if not isinstance(multigrid, list) and not isinstance(multigrid, Multigrid):
            multigrid = [[2, 2, 2, 2], [4, 4, 4, 4]]

    if clover_csw != 0.0:
        return fermion.CloverWilsonDirac(latt_info, mass, tol, maxiter, clover_csw, clover_xi, multigrid)
    else:
        return fermion.WilsonDirac(latt_info, mass, tol, maxiter, multigrid)


def getStaggeredDirac(
    latt_info: LatticeInfo,
    mass: float,
    tol: float,
    maxiter: int,
    tadpole_coeff: float = 1.0,
    naik_epsilon: float = 0.0,
):
    assert latt_info.anisotropy == 1.0

    return fermion.HISQDirac(latt_info, mass, tol, maxiter, naik_epsilon, None)


def getWilson(
    latt_info: LatticeInfo,
    mass: float,
    tol: float,
    maxiter: int,
    multigrid: List[List[int]] = None,
):
    if not multigrid:
        multigrid = None
    else:
        if not isinstance(multigrid, list) and not isinstance(multigrid, Multigrid):
            multigrid = [[2, 2, 2, 2], [4, 4, 4, 4]]

    return fermion.WilsonDirac(latt_info, mass, tol, maxiter, multigrid)


def getClover(
    latt_info: LatticeInfo,
    mass: float,
    tol: float,
    maxiter: int,
    xi_0: float = 1.0,
    clover_csw_t: float = 0.0,
    clover_csw_r: float = 1.0,
    multigrid: List[List[int]] = None,
):
    assert clover_csw_t != 0.0
    xi = latt_info.anisotropy
    if xi != 1.0:
        clover_csw = xi_0 * clover_csw_t**2 / clover_csw_r
        clover_xi = (xi_0 * clover_csw_t / clover_csw_r) ** 0.5
    else:
        clover_csw = clover_csw_t
        clover_xi = 1.0
    if not multigrid:
        multigrid = None
    else:
        if not isinstance(multigrid, list) and not isinstance(multigrid, Multigrid):
            multigrid = [[2, 2, 2, 2], [4, 4, 4, 4]]

    return fermion.CloverWilsonDirac(latt_info, mass, tol, maxiter, clover_csw, clover_xi, multigrid)


def getStaggered(
    latt_info: LatticeInfo,
    mass: float,
    tol: float,
    maxiter: int,
    tadpole_coeff: float = 1.0,
):
    assert latt_info.anisotropy == 1.0

    return fermion.StaggeredDirac(latt_info, mass, tol, maxiter, tadpole_coeff, None)


def getHISQ(
    latt_info: LatticeInfo,
    mass: float,
    tol: float,
    maxiter: int,
    naik_epsilon: float = 0.0,
    multigrid: Union[List[List[int]], Multigrid] = None,
):
    assert latt_info.anisotropy == 1.0

    return fermion.HISQDirac(latt_info, mass, tol, maxiter, naik_epsilon, multigrid)


def getDefaultDirac(
    mass: float,
    tol: float,
    maxiter: int,
    xi_0: float = 1.0,
    clover_coeff_t: float = 0.0,
    clover_coeff_r: float = 1.0,
    multigrid: Union[List[List[int]], Multigrid] = None,
):
    return getDirac(getDefaultLattice(), mass, tol, maxiter, xi_0, clover_coeff_t, clover_coeff_r, multigrid)


def getDefaultStaggeredDirac(
    mass: float,
    tol: float,
    maxiter: int,
    tadpole_coeff: float = 1.0,
    naik_epsilon: float = 0.0,
):
    return getStaggeredDirac(getDefaultLattice(), mass, tol, maxiter, tadpole_coeff, naik_epsilon)
