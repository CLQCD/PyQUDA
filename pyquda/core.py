from typing import List, Union
from enum import IntEnum
from math import sqrt
import numpy as np
import cupy as cp

from .pyquda import Pointer, getDataPointers, getDataPointer, getEvenPointer, getOddPointer


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
    def __init__(self, latt_size: List[int], value=None) -> None:
        self.latt_size = latt_size
        if value is None:
            self.data = newLatticeFieldData(latt_size, "Gauge").reshape(-1)
        else:
            self.data = value.reshape(-1)

    def setAntiPeroidicT(self):
        Lt = self.latt_size[Nd - 1]
        data = self.data.reshape(Nd, 2, Lt, -1)
        data[Nd - 1, :, Lt - 1] *= -1

    def setAnisotropy(self, anisotropy: float):
        data = self.data.reshape(Nd, -1)
        data[:Nd - 1] /= anisotropy

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


def source(latt_size: List[int], source_type: str, t_srce: Union[int, List[int]], spin: int, color: int, phase=None):
    Lx, Ly, Lz, Lt = latt_size
    b = LatticeFermion(latt_size)
    data = b.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    if source_type.lower() == "point":
        x, y, z, t = t_srce
        eo = (x + y + z + t) % 2
        data[eo, t, z, y, x // 2, spin, color] = 1
    elif source_type.lower() == "wall":
        t = t_srce
        data[:, t, :, :, :, spin, color] = 1
    elif source_type.lower() == "momentum":
        t = t_srce
        data[:, t, :, :, :, spin, color] = phase[:, t, :, :, :]
    else:
        raise NotImplementedError(f"{source_type} source is not implemented yet.")

    return b


class QudaFieldLoader:
    def __init__(
        self,
        latt_size,
        mass,
        tol,
        maxiter,
        xi_0: float = 1.0,
        nu: float = 1.0,
        clover_coeff_t: float = 0.0,
        clover_coeff_r: float = 1.0,
    ) -> None:

        Lx, Ly, Lz, Lt = latt_size
        volume = Lx * Ly * Lz * Lt
        xi = xi_0 / nu
        kappa = 1 / (2 * (mass + 1 + (Nd - 1) / xi))
        if xi != 1.0:
            clover_coeff = xi_0 * clover_coeff_t**2 / clover_coeff_r
            clover_xi = sqrt(xi_0 * clover_coeff_t / clover_coeff_r)
        else:
            clover_coeff = clover_coeff_t
            clover_xi = 1.0
        clover = clover_coeff != 0.0

        self.latt_size = latt_size
        self.volume = volume
        self.xi_0 = xi_0
        self.nu = nu
        self.xi = xi
        self.mass = mass
        self.kappa = kappa
        self.clover_coeff = kappa * clover_coeff
        self.clover_xi = clover_xi
        self.clover = clover
        if not clover:
            from .dslash import wilson as loader
            self.invert_param = loader.newQudaInvertParam(kappa, tol, maxiter)
        else:
            from .dslash import clover_wilson as loader
            self.invert_param = loader.newQudaInvertParam(kappa, tol, maxiter, clover_xi, clover_coeff)
        self.loader = loader
        self.gauge_param = loader.newQudaGaugeParam(latt_size, xi)

    def loadGauge(self, gauge: LatticeGauge):
        self.loader.loadGauge(gauge, self.gauge_param)

    def invert(self, b: LatticeFermion):
        return self.loader.invert(b, self.invert_param)
