from typing import List
from enum import IntEnum
from math import sqrt

import numpy as np
import cupy as cp

from . import pyquda as quda
from . import enum_quda
from .pyquda import getDataPointers, getDataPointer, getEvenPointer, getOddPointer


class LatticeConstant(IntEnum):
    Nc = 3
    Nd = 4
    Ns = 4


Nc = LatticeConstant.Nc
Nd = LatticeConstant.Nd
Ns = LatticeConstant.Ns


def newLatticeFieldData(latt_size: List[int], dtype: str) -> cp.ndarray:
    Lx, Ly, Lz, Lt = latt_size
    if dtype.capitalize() == "Gauge":
        return cp.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
    elif dtype.capitalize() == "Fermion":
        return cp.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c16")
    elif dtype.capitalize() == "Propagator":
        return cp.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), "<c16")


class LatticeField:
    def __init__(self) -> None:
        pass


class LatticeGauge(LatticeField):
    def __init__(self, latt_size: List[int], value=None, t_boundary=True) -> None:
        self.latt_size = latt_size
        if value is None:
            self.data = newLatticeFieldData(latt_size, "Gauge").reshape(-1)
        else:
            self.data = value.reshape(-1)
        self.t_boundary = t_boundary

    def copy(self):
        res = LatticeGauge(self.latt_size)
        res.data[:] = self.data[:]
        return res

    def setAntiPeroidicT(self):
        if self.t_boundary:
            Lt = self.latt_size[Nd - 1]
            data = self.data.reshape(Nd, 2, Lt, -1)
            data[Nd - 1, :, Lt - 1] *= -1

    def setAnisotropy(self, anisotropy: float):
        data = self.data.reshape(Nd, -1)
        data[:Nd - 1] /= anisotropy

    def lexico(self):
        Lx, Ly, Lz, Lt = self.latt_size
        data_cb2 = self.data.reshape(Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc).get()
        data_lex = np.zeros((Nd, Lt, Lz, Ly, Lx, Nc, Nc), "<c16")
        for t in range(Lt):
            for z in range(Lz):
                for y in range(Ly):
                    eo = (t + z + y) % 2
                    if eo == 0:
                        data_lex[:, t, z, y, 0::2] = data_cb2[:, 0, t, z, y, :]
                        data_lex[:, t, z, y, 1::2] = data_cb2[:, 1, t, z, y, :]
                    else:
                        data_lex[:, t, z, y, 1::2] = data_cb2[:, 0, t, z, y, :]
                        data_lex[:, t, z, y, 0::2] = data_cb2[:, 1, t, z, y, :]
        return data_lex.reshape(-1)

    @property
    def data_ptr(self):
        return getDataPointers(self.data.reshape(4, -1), 4)

    @property
    def data_ptrs(self):
        return getDataPointers(self.data.reshape(4, -1), 4)


class LatticeFermion(LatticeField):
    def __init__(self, latt_size: List[int]) -> None:
        self.latt_size = latt_size
        self.data = newLatticeFieldData(latt_size, "Fermion").reshape(-1)

    @property
    def even(self):
        return self.data.reshape(2, -1)[0]

    @even.setter
    def even(self, value):
        data = self.data.reshape(2, -1)
        data[0] = value.reshape(-1)

    @property
    def odd(self):
        return self.data.reshape(2, -1)[1]

    @odd.setter
    def odd(self, value):
        data = self.data.reshape(2, -1)
        data[1] = value.reshape(-1)

    @property
    def data_ptr(self):
        return getDataPointer(self.data)

    @property
    def even_ptr(self):
        return getEvenPointer(self.data.reshape(2, -1))

    @property
    def odd_ptr(self):
        return getOddPointer(self.data.reshape(2, -1))


class LatticePropagator(LatticeField):
    def __init__(self, latt_size: List[int]) -> None:
        self.latt_size = latt_size
        self.data = newLatticeFieldData(latt_size, "Propagator").reshape(-1)

    def lexico(self):
        Lx, Ly, Lz, Lt = self.latt_size
        data_cb2 = self.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc).get()
        data_lex = np.zeros((Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), "<c16")
        for t in range(Lt):
            for z in range(Lz):
                for y in range(Ly):
                    eo = (t + z + y) % 2
                    if eo == 0:
                        data_lex[t, z, y, 0::2] = data_cb2[0, t, z, y, :]
                        data_lex[t, z, y, 1::2] = data_cb2[1, t, z, y, :]
                    else:
                        data_lex[t, z, y, 1::2] = data_cb2[0, t, z, y, :]
                        data_lex[t, z, y, 0::2] = data_cb2[1, t, z, y, :]
        return data_lex.reshape(-1)

    def transpose(self):
        Lx, Ly, Lz, Lt = self.latt_size
        Vol = Lx * Ly * Lz * Lt
        data = self.data.reshape(Vol, Ns, Ns, Nc, Nc)
        data_T = data.transpose(0, 2, 1, 4, 3).copy()
        return data_T.reshape(-1)


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
    quda.saveGaugeQuda(gauge.data_ptr, dslash.gauge_param)


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
    quda.saveGaugeQuda(gauge.data_ptr, dslash.gauge_param)


def invert12(b12: LatticePropagator, dslash):
    latt_size = b12.latt_size
    Lx, Ly, Lz, Lt = latt_size
    Vol = Lx * Ly * Lz * Lt

    x12 = LatticePropagator(latt_size)
    for spin in range(Ns):
        for color in range(Nc):
            b = LatticeFermion(latt_size)
            data = b.data.reshape(Vol, Ns, Nc)
            data[:] = b12.data.reshape(Vol, Ns, Ns, Nc, Nc)[:, :, spin, :, color]
            x = dslash.invert(b)
            data = x12.data.reshape(Vol, Ns, Ns, Nc, Nc)
            data[:, :, spin, :, color] = x.data.reshape(Vol, Ns, Nc)

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
    xi = xi_0 / nu
    kappa = 1 / (2 * (mass + 1 + (Nd - 1) / xi))
    if xi != 1.0:
        clover_coeff = xi_0 * clover_coeff_t**2 / clover_coeff_r
        clover_xi = sqrt(xi_0 * clover_coeff_t / clover_coeff_r)
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

    if clover_coeff != 0.0:
        from .dslash import clover_wilson
        return clover_wilson.CloverWilson(
            latt_size, kappa, tol, maxiter, xi, clover_coeff, clover_xi, t_boundary, geo_block_size
        )
    else:
        from .dslash import wilson
        return wilson.Wilson(latt_size, kappa, tol, maxiter, xi, t_boundary, geo_block_size)
