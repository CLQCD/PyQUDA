from abc import ABC
from typing import List, Union

from . import getCUDABackend
from .pointer import Pointers
from .pyquda import (
    setVerbosityQuda,
    gaussGaugeQuda,
    gaussMomQuda,
    loadGaugeQuda,
    saveGaugeQuda,
    momResidentQuda,
    momActionQuda,
    plaqQuda,
    updateGaugeFieldQuda,
)
from .enum_quda import QudaTboundary, QudaVerbosity
from .field import Nc, Ns, LatticeInfo, LatticeGauge, LatticeMom, LatticeFermion
from .dirac.wilson import Wilson
from .action import FermionAction, GaugeAction

nullptr = Pointers("void", 0)


class Integrator(ABC):
    @classmethod
    def integrate(cls, hmc: "HMC", t: float, n_steps: int):
        raise NotImplementedError


class O2Nf1Ng0V(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    BAB: Eq.(23), \xi=0"""

    @classmethod
    def integrate(cls, hmc: "HMC", t: float, n_steps: int):
        dt = t / n_steps
        for _ in range(n_steps):
            hmc.updateMom(dt / 2)
            hmc.updateGauge(dt)
            hmc.updateMom(dt / 2)


class O2Nf1Ng0P(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    ABA: Eq.(24), \xi=0"""

    @classmethod
    def integrate(cls, hmc: "HMC", t: float, n_steps: int):
        dt = t / n_steps
        for _ in range(n_steps):
            hmc.updateGauge(dt / 2)
            hmc.updateMom(dt)
            hmc.updateGauge(dt / 2)


class O4Nf5Ng0V(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    BABABABABAB: Eq.(63), Eq.(71)"""

    rho_ = 0.2539785108410595
    theta_ = -0.03230286765269967
    vartheta_ = 0.08398315262876693
    lambda_ = 0.6822365335719091

    @classmethod
    def integrate(cls, hmc: "HMC", t: float, n_steps: int):
        dt = t / n_steps
        for _ in range(n_steps):
            hmc.updateMom(cls.vartheta_ * dt)
            hmc.updateGauge(cls.rho_ * dt)
            hmc.updateMom(cls.lambda_ * dt)
            hmc.updateGauge(cls.theta_ * dt)
            hmc.updateMom((1 - 2 * (cls.lambda_ + cls.vartheta_)) * dt / 2)
            hmc.updateGauge((1 - 2 * (cls.theta_ + cls.rho_)) * dt)
            hmc.updateMom((1 - 2 * (cls.lambda_ + cls.vartheta_)) * dt / 2)
            hmc.updateGauge(cls.theta_ * dt)
            hmc.updateMom(cls.lambda_ * dt)
            hmc.updateGauge(cls.rho_ * dt)
            hmc.updateMom(cls.vartheta_ * dt)


class O4Nf5Ng0P(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    ABABABABABA: Eq.(72), Eq.(80)"""

    rho_ = 0.2750081212332419
    theta_ = -0.1347950099106792
    vartheta_ = -0.08442961950707149
    lambda_ = 0.3549000571574260

    @classmethod
    def integrate(cls, hmc: "HMC", t: float, n_steps: int):
        dt = t / n_steps
        for _ in range(n_steps):
            hmc.updateGauge(cls.rho_ * dt)
            hmc.updateMom(cls.vartheta_ * dt)
            hmc.updateGauge(cls.theta_ * dt)
            hmc.updateMom(cls.lambda_ * dt)
            hmc.updateGauge((1 - 2 * (cls.theta_ + cls.rho_)) * dt / 2)
            hmc.updateMom((1 - 2 * (cls.lambda_ + cls.vartheta_)) * dt)
            hmc.updateGauge((1 - 2 * (cls.theta_ + cls.rho_)) * dt / 2)
            hmc.updateMom(cls.lambda_ * dt)
            hmc.updateGauge(cls.theta_ * dt)
            hmc.updateMom(cls.vartheta_ * dt)
            hmc.updateGauge(cls.rho_ * dt)


class HMC:
    def __init__(
        self, latt_info: LatticeInfo, monomials: List[Union[GaugeAction, FermionAction]], integrator: Integrator
    ) -> None:
        self.latt_info = latt_info
        self._monomials = monomials
        self._integrator = integrator
        self._wilson = Wilson(latt_info, 0, 0.125, 0, 0)
        self.gauge_param = self._wilson.gauge_param

    def setVerbosity(self, verbosity: QudaVerbosity):
        setVerbosityQuda(verbosity, b"\0")

    def initialize(self, gauge: Union[LatticeGauge, int, None] = None):
        if isinstance(gauge, LatticeGauge):
            self.loadGauge(gauge)
        else:
            unit = LatticeGauge(self.latt_info)
            self.loadGauge(unit)
            if gauge is not None:
                self.gaussGauge(gauge)
        mom = LatticeMom(self.latt_info)
        self.loadMom(mom)

    def actionGauge(self) -> float:
        action = 0
        for monomial in self._monomials:
            if isinstance(monomial, FermionAction):
                action += monomial.action(True)
            elif isinstance(monomial, GaugeAction):
                action += monomial.action()
        return action

    def actionMom(self) -> float:
        return momActionQuda(nullptr, self.gauge_param)

    def updateGauge(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, True, self.gauge_param)
        loadGaugeQuda(nullptr, self.gauge_param)

    def updateMom(self, dt: float):
        for monomial in self._monomials:
            if isinstance(monomial, FermionAction):
                monomial.force(dt, True)
            elif isinstance(monomial, GaugeAction):
                monomial.force(dt)

    def integrate(self, t: float, n_steps: int):
        self._integrator.integrate(self, t, n_steps)

    def samplePhi(self, seed: int):
        def _seed(seed: int):
            backend = getCUDABackend()
            if backend == "numpy":
                import numpy

                numpy.random.seed(seed)
                return numpy, numpy.random.random, numpy.float64
            elif backend == "cupy":
                import cupy

                cupy.random.seed(seed)
                return cupy, cupy.random.random, cupy.float64
            elif backend == "torch":
                import torch

                torch.random.manual_seed(seed)
                return torch, torch.rand, torch.float64

        def _noise(backend, random, float64):
            phi = 2 * backend.pi * random((self.latt_info.volume, Ns, Nc), dtype=float64)
            r = random((self.latt_info.volume, Ns, Nc), dtype=float64)
            noise_raw = backend.sqrt(-backend.log(r)) * (backend.cos(phi) + 1j * backend.sin(phi))
            return noise_raw

        backend, random, float64 = _seed(seed)
        for monomial in self._monomials:
            if isinstance(monomial, FermionAction):
                monomial.sample(LatticeFermion(self.latt_info, _noise(backend, random, float64)), True)

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeriodicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeriodicT()

    def gaussGauge(self, seed: int):
        gaussGaugeQuda(seed, 1.0)

    def loadMom(self, mom: LatticeMom):
        momResidentQuda(mom.data_ptrs, self.gauge_param)

    def saveMom(self, mom: LatticeMom):
        self.gauge_param.make_resident_mom = 0
        self.gauge_param.return_result_mom = 1
        momResidentQuda(mom.data_ptrs, self.gauge_param)
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0
        momResidentQuda(mom.data_ptrs, self.gauge_param)  # keep momResident

    def gaussMom(self, seed: int):
        gaussMomQuda(seed, 1.0)

    def reunitGauge(self, tol: float):
        gauge = LatticeGauge(self.latt_info)
        self.saveGauge(gauge)
        gauge.projectSU3(tol)
        self.loadGauge(gauge)

    def plaquette(self):
        return plaqQuda()[0]
