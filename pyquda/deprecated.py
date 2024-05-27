from typing import List

from . import getLogger, getGridSize, quda, enum_quda
from .field import LatticeFermion, LatticeGauge, LatticeInfo, LatticePropagator, Nc, Nd, Ns
from .dirac import Dirac


def smear(latt_size: List[int], gauge: LatticeGauge, nstep: int, rho: float):
    getLogger().warning("Use GaugeField.stoutSmear instead", DeprecationWarning)
    from .core import getDslash

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
    getLogger().warning("Use GaugeField.stoutSmear instead", DeprecationWarning)
    from .core import getDslash

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


def invert12(b12: LatticePropagator, dslash: Dirac):
    getLogger().warning("Use core.invert instead", DeprecationWarning)
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
    getLogger().warning("Use getDirac instead", DeprecationWarning)
    Gx, Gy, Gz, Gt = getGridSize()
    Lx, Ly, Lz, Lt = latt_size
    Lx, Ly, Lz, Lt = Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt

    xi = xi_0 / nu
    kappa = 1 / (2 * (mass + 1 + (Nd - 1) / xi))
    if xi != 1.0:
        clover_csw = xi_0 * clover_coeff_t**2 / clover_coeff_r
        clover_xi = (xi_0 * clover_coeff_t / clover_coeff_r) ** 0.5
    else:
        clover_csw = clover_coeff_t
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

    if clover_csw != 0.0:
        from .dirac.clover_wilson import CloverWilson

        return CloverWilson(latt_info, mass, kappa, tol, maxiter, clover_csw, clover_xi, geo_block_size)
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
    getLogger().warning("Use getStaggeredDirac instead", DeprecationWarning)
    Gx, Gy, Gz, Gt = getGridSize()
    Lx, Ly, Lz, Lt = latt_size
    Lx, Ly, Lz, Lt = Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt

    kappa = 1 / 2
    if anti_periodic_t:
        t_boundary = -1
    else:
        t_boundary = 1
    latt_info = LatticeInfo([Lx, Ly, Lz, Lt], t_boundary, 1.0)

    from .dirac.hisq import HISQ

    return HISQ(latt_info, mass, kappa, tol, maxiter, tadpole_coeff, naik_epsilon, None)
