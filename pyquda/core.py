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
from .dirac import Dirac
from .utils.source import source

_DEFAULT_LATTICE: LatticeInfo = None


def setDefaultLattice(latt_size: List[int], t_boundary: Literal[1, -1] = -1, anisotropy: float = 1.0):
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


def smear(latt_size: List[int], gauge: LatticeGauge, nstep: int, rho: float):
    smear_param = quda.QudaGaugeSmearParam()
    smear_param.n_steps = nstep
    smear_param.rho = rho
    smear_param.meas_interval = nstep + 1
    smear_param.smear_type = enum_quda.QudaGaugeSmearType.QUDA_GAUGE_SMEAR_STOUT
    obs_param = quda.QudaGaugeObservableParam()
    obs_param.compute_qcharge = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
    dslash = getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
    dslash.gauge_param.reconstruct = enum_quda.QudaReconstructType.QUDA_RECONSTRUCT_NO
    dslash.loadGauge(gauge)
    quda.performGaugeSmearQuda(smear_param, obs_param)
    dslash.gauge_param.type = enum_quda.QudaLinkType.QUDA_SMEARED_LINKS
    quda.saveGaugeQuda(gauge.data_ptrs, dslash.gauge_param)


def smear4(latt_size: List[int], gauge: LatticeGauge, nstep: int, rho: float):
    smear_param = quda.QudaGaugeSmearParam()
    smear_param.n_steps = nstep
    smear_param.rho = rho
    smear_param.epsilon = 1.0
    smear_param.meas_interval = nstep + 1
    smear_param.smear_type = enum_quda.QudaGaugeSmearType.QUDA_GAUGE_SMEAR_OVRIMP_STOUT
    obs_param = quda.QudaGaugeObservableParam()
    obs_param.compute_qcharge = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
    dslash = getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
    dslash.gauge_param.reconstruct = enum_quda.QudaReconstructType.QUDA_RECONSTRUCT_NO
    dslash.loadGauge(gauge)
    quda.performGaugeSmearQuda(smear_param, obs_param)
    dslash.gauge_param.type = enum_quda.QudaLinkType.QUDA_SMEARED_LINKS
    quda.saveGaugeQuda(gauge.data_ptrs, dslash.gauge_param)


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
    dslash: Dirac,
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


def invert12(b12: LatticePropagator, dslash: Dirac):
    latt_info = b12.latt_info
    Vol = latt_info.volume

    x12 = LatticePropagator(latt_info)
    for spin in range(Ns):
        for color in range(Nc):
            b = LatticeFermion(latt_info)
            data = b.data.reshape(Vol, Ns, Nc)
            data[:] = b12.data.reshape(Vol, Ns, Ns, Nc, Nc)[:, :, spin, :, color]
            x = dslash.invert(b)
            data = x12.data.reshape(Vol, Ns, Ns, Nc, Nc)
            data[:, :, spin, :, color] = x.data.reshape(Vol, Ns, Nc)
            b = None

    return x12


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
    prefix_size = numpy.prod(prefix)
    suffix_size = numpy.prod(suffix)

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


def getDslash(
    latt_size: List[int],
    mass: float,
    tol: float,
    maxiter: int,
    xi_0: float = 1.0,
    nu: float = 1.0,
    clover_coeff_t: float = 0.0,
    clover_coeff_r: float = 1.0,
    anti_periodic_t: bool = True,
    multigrid: List[List[int]] = None,
):
    Gx, Gy, Gz, Gt = getGridSize()
    Lx, Ly, Lz, Lt = latt_size
    Lx, Ly, Lz, Lt = Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt

    xi = xi_0 / nu
    kappa = 1 / (2 * (mass + 1 + (Nd - 1) / xi))
    if xi != 1.0:
        clover_coeff = xi_0 * clover_coeff_t**2 / clover_coeff_r
        clover_xi = (xi_0 * clover_coeff_t / clover_coeff_r) ** 0.5
    else:
        clover_coeff = clover_coeff_t
        clover_xi = 1.0
    if anti_periodic_t:
        t_boundary = -1
    else:
        t_boundary = 1
    if not multigrid:
        geo_block_size = None
    else:
        if not isinstance(multigrid, list):
            geo_block_size = [[2, 2, 2, 2], [4, 4, 4, 4]]
        else:
            geo_block_size = multigrid
    latt_info = LatticeInfo([Lx, Ly, Lz, Lt], t_boundary, xi)

    if clover_coeff != 0.0:
        from .dirac.clover_wilson import CloverWilson

        return CloverWilson(latt_info, mass, kappa, tol, maxiter, clover_coeff, clover_xi, geo_block_size)
    else:
        from .dirac.wilson import Wilson

        return Wilson(latt_info, mass, kappa, tol, maxiter, geo_block_size)


def getStaggeredDslash(
    latt_size: List[int],
    mass: float,
    tol: float,
    maxiter: int,
    tadpole_coeff: float = 1.0,
    naik_epsilon: float = 0.0,
    anti_periodic_t: bool = True,
):
    Gx, Gy, Gz, Gt = getGridSize()
    Lx, Ly, Lz, Lt = latt_size
    Lx, Ly, Lz, Lt = Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt

    kappa = 1 / (2 * (mass + Nd))
    if anti_periodic_t:
        t_boundary = -1
    else:
        t_boundary = 1
    latt_info = LatticeInfo([Lx, Ly, Lz, Lt], t_boundary, 1.0)

    from .dirac.hisq import HISQ

    return HISQ(latt_info, mass, kappa, tol, maxiter, tadpole_coeff, naik_epsilon, None)


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
    kappa = 1 / (2 * (mass + Nd))

    from .dirac.hisq import HISQ

    return HISQ(latt_info, mass, kappa, tol, maxiter, tadpole_coeff, naik_epsilon, None)
