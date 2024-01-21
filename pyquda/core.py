from typing import List, Literal, Union

import numpy

from . import pyquda as quda, enum_quda, getMPIComm, getMPISize, getMPIRank, getGridSize, getCoordFromRank
from .field import (
    Ns,
    Nc,
    Nd,
    LatticeInfo,
    LatticeGauge,
    LatticeFermion,
    LatticePropagator,
    LatticeStaggeredFermion,
    LatticeStaggeredPropagator,
    lexico,
    cb2,
)
from .dirac import Dirac, StaggeredDirac
from .utils.source import source
from .deprecated import smear, smear4, invert12, getDslash, getStaggeredDslash

_DEFAULT_LATTICE: LatticeInfo = None


def setDefaultLattice(latt_size: List[int], t_boundary: Literal[1, -1], anisotropy: float):
    global _DEFAULT_LATTICE
    _DEFAULT_LATTICE = LatticeInfo(latt_size, t_boundary, anisotropy)


def getDefaultLattice():
    return _DEFAULT_LATTICE


class LatticeGaugeDefault(LatticeGauge):
    def __init__(self, value=None) -> None:
        super().__init__(_DEFAULT_LATTICE, value)


class LatticeFermionDefault(LatticeFermion):
    def __init__(self, value=None) -> None:
        super().__init__(_DEFAULT_LATTICE, value)


class LatticePropagatorDefault(LatticePropagator):
    def __init__(self, value=None) -> None:
        super().__init__(_DEFAULT_LATTICE, value)


class LatticeStaggeredFermionDefault(LatticeStaggeredFermion):
    def __init__(self, value=None) -> None:
        super().__init__(_DEFAULT_LATTICE, value)


class LatticeStaggeredPropagatorDefault(LatticeStaggeredPropagator):
    def __init__(self, value=None) -> None:
        super().__init__(_DEFAULT_LATTICE, value)


def invert(
    dslash: Dirac,
    source_type: str,
    t_srce: Union[int, List[int]],
    source_phase=None,
    rho: float = 0.0,
    nsteps: int = 1,
):
    latt_info = dslash.latt_info
    Vol = latt_info.volume
    xi = dslash.gauge_param.anisotropy

    prop = LatticePropagator(latt_info)
    data = prop.data.reshape(Vol, Ns, Ns, Nc, Nc)
    for spin in range(Ns):
        for color in range(Nc):
            b = source(latt_info.size, source_type, t_srce, spin, color, source_phase, rho, nsteps, xi)
            x = dslash.invert(b)
            data[:, :, spin, :, color] = x.data.reshape(Vol, Ns, Nc)

    return prop


def invertStaggered(
    dslash: StaggeredDirac,
    source_type: str,
    t_srce: Union[int, List[int]],
    source_phase=None,
    rho: float = 0.0,
    nsteps: int = 1,
):
    latt_info = dslash.latt_info
    Vol = latt_info.volume
    xi = dslash.latt_info.anisotropy

    prop = LatticeStaggeredPropagator(latt_info)
    data = prop.data.reshape(Vol, Nc, Nc)
    for color in range(Nc):
        b = source(latt_info.size, source_type, t_srce, None, color, source_phase, rho, nsteps, xi)
        x = dslash.invert(b)
        data[:, :, color] = x.data.reshape(Vol, Nc)

    return prop


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
    prefix_size = int(numpy.prod(prefix))
    suffix_size = int(numpy.prod(suffix))

    if getMPIRank() == root:
        sendbuf = data.reshape(-1)
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
            raise NotImplementedError(f"core.gather doesn't support reduce operator reduce_op={reduce_op}")
    else:
        sendbuf = data.reshape(-1)
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
    multigrid: List[List[int]] = None,
):
    xi = latt_info.anisotropy
    kappa = 1 / (2 * (mass + 1 + (Nd - 1) / xi))
    if xi != 1.0:
        clover_coeff = xi_0 * clover_coeff_t**2 / clover_coeff_r
        clover_xi = (xi_0 * clover_coeff_t / clover_coeff_r) ** 0.5
    else:
        clover_coeff = clover_coeff_t
        clover_xi = 1.0
    if not multigrid:
        geo_block_size = None
    else:
        if not isinstance(multigrid, list):
            geo_block_size = [[2, 2, 2, 2], [4, 4, 4, 4]]
        else:
            geo_block_size = multigrid

    if clover_coeff != 0.0:
        from .dirac.clover_wilson import CloverWilson

        return CloverWilson(latt_info, mass, kappa, tol, maxiter, clover_coeff, clover_xi, geo_block_size)
    else:
        from .dirac.wilson import Wilson

        return Wilson(latt_info, mass, kappa, tol, maxiter, geo_block_size)


def getStaggeredDirac(
    latt_info: LatticeInfo,
    mass: float,
    tol: float,
    maxiter: int,
    tadpole_coeff: float = 1.0,
    naik_epsilon: float = 0.0,
):
    assert latt_info.anisotropy == 1.0
    kappa = 1 / 2

    from .dirac.hisq import HISQ

    return HISQ(latt_info, mass, kappa, tol, maxiter, tadpole_coeff, naik_epsilon, None)


def getDiracDefault(
    mass: float,
    tol: float,
    maxiter: int,
    xi_0: float = 1.0,
    clover_coeff_t: float = 0.0,
    clover_coeff_r: float = 1.0,
    multigrid: List[List[int]] = None,
):
    assert _DEFAULT_LATTICE is not None
    return getDirac(_DEFAULT_LATTICE, mass, tol, maxiter, xi_0, clover_coeff_t, clover_coeff_r, multigrid)


def getStaggeredDiracDefault(
    mass: float,
    tol: float,
    maxiter: int,
    tadpole_coeff: float = 1.0,
    naik_epsilon: float = 0.0,
):
    assert _DEFAULT_LATTICE is not None
    return getStaggeredDirac(_DEFAULT_LATTICE, mass, tol, maxiter, tadpole_coeff, naik_epsilon)
