from abc import ABC, abstractmethod
from math import exp
from random import Random
from typing import List, Union

from . import getLogger, getCUDABackend
from .pointer import Pointers
from .pyquda import (
    QudaGaugeObservableParam,
    gaugeObservablesQuda,
    gaussMomQuda,
    loadGaugeQuda,
    saveGaugeQuda,
    momResidentQuda,
    momActionQuda,
    updateGaugeFieldQuda,
)
from .enum_quda import QudaBoolean, QudaTboundary
from .field import (
    Nc,
    Ns,
    LatticeInfo,
    LatticeGauge,
    LatticeMom,
    LatticeFermion,
    LatticeStaggeredFermion,
    LatticeFloat64,
)
from .dirac.wilson import Wilson
from .dirac.staggered import Staggered
from .action.abstract import GaugeAction, FermionAction, StaggeredFermionAction

nullptr = Pointers("void", 0)


class Integrator(ABC):
    def __init__(self, n_steps: int) -> None:
        self.n_steps = n_steps

    @abstractmethod
    def integrate(self, updateGauge, updateMom, t: float):
        raise NotImplementedError("The integrator is not implemented yet")


class INT_3G1F(Integrator):
    alpha_ = 0.1
    beta_ = 0.1

    def integrate(self, updateGauge, updateMom, t: float):
        assert self.n_steps % 2 == 0
        dt = t / self.n_steps
        for _ in range(0, self.n_steps, 2):
            updateGauge(dt * (1 / 6 - self.alpha_ / 3))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (1 / 3 + self.alpha_ / 3 - self.beta_))
            updateMom(dt, gauge=False)
            updateGauge(dt * (self.alpha_ / 3 + self.beta_))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (1 / 3 - self.alpha_ * 2 / 3))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (1 / 3 + self.alpha_ * 2 / 3))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (1 / 3 - self.alpha_ * 2 / 3))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (self.alpha_ / 3 + self.beta_))
            updateMom(dt, gauge=False)
            updateGauge(dt * (1 / 3 + self.alpha_ / 3 - self.beta_))
            updateMom(dt / 3, fermion=False)
            updateGauge(dt * (1 / 6 - self.alpha_ / 3))


class O2Nf1Ng0V(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    BAB: Eq.(23), \xi=0"""

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateMom(dt / 2)
        for _ in range(self.n_steps - 1):
            updateGauge(dt)
            updateMom(dt)
        updateGauge(dt)
        updateMom(dt / 2)


class O2Nf1Ng0P(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    ABA: Eq.(24), \xi=0"""

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateGauge(dt / 2)
        for _ in range(self.n_steps - 1):
            updateMom(dt)
            updateGauge(dt)
        updateMom(dt)
        updateGauge(dt / 2)


class O2Nf2Ng0V(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    BABAB: Eq.(25), Eq.(31)"""

    lambda_ = 0.1931833275037836

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateMom(self.lambda_ * dt)
        for _ in range(self.n_steps - 1):
            updateGauge(dt / 2)
            updateMom((1 - 2 * self.lambda_) * dt)
            updateGauge(dt / 2)
            updateMom(2 * self.lambda_ * dt)
        updateGauge(dt / 2)
        updateMom((1 - 2 * self.lambda_) * dt)
        updateGauge(dt / 2)
        updateMom(self.lambda_ * dt)


class O2Nf2Ng0P(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    ABABA: Eq.(32), Eq.(31)"""

    lambda_ = 0.1931833275037836

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateGauge(self.lambda_ * dt)
        for _ in range(self.n_steps - 1):
            updateMom(dt / 2)
            updateGauge((1 - 2 * self.lambda_) * dt)
            updateMom(dt / 2)
            updateGauge(2 * self.lambda_ * dt)
        updateMom(dt / 2)
        updateGauge((1 - 2 * self.lambda_) * dt)
        updateMom(dt / 2)
        updateGauge(self.lambda_ * dt)


class O4Nf5Ng0V(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    BABABABABAB: Eq.(63), Eq.(71)"""

    rho_ = 0.2539785108410595
    theta_ = -0.03230286765269967
    vartheta_ = 0.08398315262876693
    lambda_ = 0.6822365335719091

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateMom(self.vartheta_ * dt)
        for _ in range(self.n_steps - 1):
            updateGauge(self.rho_ * dt)
            updateMom(self.lambda_ * dt)
            updateGauge(self.theta_ * dt)
            updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt / 2)
            updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt)
            updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt / 2)
            updateGauge(self.theta_ * dt)
            updateMom(self.lambda_ * dt)
            updateGauge(self.rho_ * dt)
            updateMom(2 * self.vartheta_ * dt)
        updateGauge(self.rho_ * dt)
        updateMom(self.lambda_ * dt)
        updateGauge(self.theta_ * dt)
        updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt / 2)
        updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt)
        updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt / 2)
        updateGauge(self.theta_ * dt)
        updateMom(self.lambda_ * dt)
        updateGauge(self.rho_ * dt)
        updateMom(self.vartheta_ * dt)


class O4Nf5Ng0P(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    ABABABABABA: Eq.(72), Eq.(80)"""

    rho_ = 0.2750081212332419
    theta_ = -0.1347950099106792
    vartheta_ = -0.08442961950707149
    lambda_ = 0.3549000571574260

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        updateGauge(self.rho_ * dt)
        for _ in range(self.n_steps - 1):
            updateMom(self.vartheta_ * dt)
            updateGauge(self.theta_ * dt)
            updateMom(self.lambda_ * dt)
            updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt / 2)
            updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt)
            updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt / 2)
            updateMom(self.lambda_ * dt)
            updateGauge(self.theta_ * dt)
            updateMom(self.vartheta_ * dt)
            updateGauge(2 * self.rho_ * dt)
        updateMom(self.vartheta_ * dt)
        updateGauge(self.theta_ * dt)
        updateMom(self.lambda_ * dt)
        updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt / 2)
        updateMom((1 - 2 * (self.lambda_ + self.vartheta_)) * dt)
        updateGauge((1 - 2 * (self.theta_ + self.rho_)) * dt / 2)
        updateMom(self.lambda_ * dt)
        updateGauge(self.theta_ * dt)
        updateMom(self.vartheta_ * dt)
        updateGauge(self.rho_ * dt)


class HMC:
    def __init__(
        self,
        latt_info: LatticeInfo,
        monomials: List[Union[GaugeAction, FermionAction, StaggeredFermionAction]],
        integrator: Integrator,
        hmc_inner: "HMC" = None,
    ) -> None:
        self.latt_info = latt_info
        self._gauge_monomials = [monomial for monomial in monomials if isinstance(monomial, GaugeAction)]
        self._fermion_monomials = [monomial for monomial in monomials if not isinstance(monomial, GaugeAction)]
        self._integrator = integrator
        self._hmc_inner = hmc_inner
        self.staggered = None
        for monomial in self._fermion_monomials:
            if self.staggered is None:
                self.staggered = isinstance(monomial, StaggeredFermionAction)
                continue
            if (isinstance(monomial, FermionAction) and self.staggered) or (
                isinstance(monomial, StaggeredFermionAction) and not self.staggered
            ):
                getLogger().critical(
                    "FermionAction and StaggeredFermionAction cannot be used at the same time", ValueError
                )
        if not self.staggered:
            self._dirac = Wilson(latt_info, 0, 0.125, 0, 0)
        else:
            self._dirac = Staggered(latt_info, 0, 0.5, 0, 0)
        self.gauge_param = self._dirac.gauge_param
        self.obs_param = QudaGaugeObservableParam()
        self.obs_param.remove_staggered_phase = QudaBoolean(not not self.staggered)
        self.fuseFermionAction()

    def fuseFermionAction(self):
        if self.staggered:
            from .action.hisq import HISQFermion, MultiHISQFermion

            hisq_monomials = [monomial for monomial in self._fermion_monomials if isinstance(monomial, HISQFermion)]
            if hisq_monomials != []:
                hisq_monomials = [MultiHISQFermion(self.latt_info, hisq_monomials)]
            self._fermion_monomials = hisq_monomials + [
                monomial for monomial in self._fermion_monomials if not isinstance(monomial, HISQFermion)
            ]

    def initializeRNG(self, seed: int):
        self.random = Random(seed * self.latt_info.volume + self.latt_info.mpi_rank)
        self.random.setstate(
            (
                self.random.VERSION,
                tuple(self.random.randrange(2**32) if i < 624 else self.random.randrange(625) for i in range(625)),
                None,
            )
        )

    def initialize(self, seed: int, gauge: LatticeGauge):
        self.initializeRNG(seed)
        self.loadGauge(gauge)
        mom = LatticeMom(self.latt_info)
        self.loadMom(mom)

    def actionGauge(self, gauge: bool = True, fermion: bool = True) -> float:
        action = 0 if self._hmc_inner is None else self._hmc_inner.actionGauge(gauge, fermion)
        if gauge:
            for monomial in self._gauge_monomials:
                action += monomial.action()
        if fermion:
            for monomial in self._fermion_monomials:
                action += monomial.action(True)
        return action

    def actionMom(self) -> float:
        return momActionQuda(nullptr, self.gauge_param)

    def updateGauge(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, True, self.gauge_param)
        loadGaugeQuda(nullptr, self.gauge_param)

    def updateMom(self, dt: float, gauge: bool = True, fermion: bool = True):
        if gauge:
            for monomial in self._gauge_monomials:
                monomial.force(dt)
        if fermion:
            for monomial in self._fermion_monomials:
                monomial.force(dt, True)

    def samplePhi(self):
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

        def _noise(backend, random, shape, float64):
            phi = 2 * backend.pi * random(shape, dtype=float64)
            r = random(shape, dtype=float64)
            noise_raw = backend.sqrt(-backend.log(r)) * (backend.cos(phi) + 1j * backend.sin(phi))
            return noise_raw

        backend, random, float64 = _seed(self.random.randrange(2**64))
        shape = (self.latt_info.volume, Ns, Nc) if not self.staggered else (self.latt_info.volume, Nc)
        for monomial in self._fermion_monomials:
            if isinstance(monomial, FermionAction):
                noise = LatticeFermion(self.latt_info, _noise(backend, random, shape, float64))
                monomial.sample(noise, True)
            elif isinstance(monomial, StaggeredFermionAction):
                if hasattr(monomial, "pseudo_fermions"):
                    noise = [
                        LatticeStaggeredFermion(self.latt_info, _noise(backend, random, shape, float64))
                        for _ in range(len(monomial.pseudo_fermions))
                    ]
                else:
                    noise = LatticeStaggeredFermion(self.latt_info, _noise(backend, random, shape, float64))
                monomial.sample(noise, True)

        if self._hmc_inner is not None:
            self._hmc_inner.samplePhi()

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.staggered:
            gauge_in.staggeredPhase()
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeriodicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeriodicT()
        if self.staggered:
            gauge.staggeredPhase(True)

    def loadMom(self, mom: LatticeMom):
        momResidentQuda(mom.data_ptrs, self.gauge_param)

    def saveMom(self, mom: LatticeMom):
        self.gauge_param.make_resident_mom = 0
        self.gauge_param.return_result_mom = 1
        momResidentQuda(mom.data_ptrs, self.gauge_param)
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0
        momResidentQuda(mom.data_ptrs, self.gauge_param)  # keep momResident

    def gaussMom(self):
        gaussMomQuda(self.random.randrange(2**64), 1.0)

    def projectSU3(self, tol: float):
        gauge = LatticeGauge(self.latt_info)
        self.saveGauge(gauge)
        gauge.projectSU3(tol)
        self.loadGauge(gauge)

    def integrate(self, t: float, project_tol: float = None):
        self._integrator.integrate(
            self.updateGauge if self._hmc_inner is None else self._hmc_inner.integrate, self.updateMom, t
        )
        if project_tol is not None:
            self.projectSU3(project_tol)

    def accept(self, delta_s):
        return self.latt_info.mpi_comm.bcast(self.random.random() < exp(-delta_s))

    def plaquette(self):
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_FALSE
        return self.obs_param.plaquette

    def polyakovLoop(self):
        self.obs_param.compute_polyakov_loop = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_polyakov_loop = QudaBoolean.QUDA_BOOLEAN_FALSE
        return self.obs_param.ploop

    def energy(self):
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE
        return self.obs_param.energy

    def qcharge(self):
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE
        return self.obs_param.qcharge

    def qchargeDensity(self):
        qcharge_density = LatticeFloat64(self.latt_info)
        self.obs_param.qcharge_density = qcharge_density.data_ptr
        self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        return qcharge_density
