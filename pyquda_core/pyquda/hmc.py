from abc import ABC, abstractmethod
from math import exp
from random import Random
from typing import List, Optional, Union

import numpy

from pyquda_comm import getLogger, getArrayBackend
from pyquda_comm.array import arrayRandomGetState, arrayRandomSetState, arrayRandomSeed
from .field import LatticeInfo, LatticeGauge, LatticeMom, LatticeReal
from .quda import (
    QudaGaugeObservableParam,
    gaugeObservablesQuda,
    gaussMomQuda,
    loadGaugeQuda,
    saveGaugeQuda,
    momResidentQuda,
    momActionQuda,
    updateGaugeFieldQuda,
)
from .enum_quda import QudaBoolean, QudaTboundary, QudaVerbosity
from .dirac import WilsonDirac, StaggeredDirac
from .action.abstract import Action, FermionAction, StaggeredFermionAction

nullptr = numpy.empty((0, 0), "<c16")


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
        monomials: List[Union[Action, FermionAction, StaggeredFermionAction]],
        integrator: Integrator,
        hmc_inner: Optional["HMC"] = None,
    ) -> None:
        self.latt_info = latt_info
        self.gauge_monomials: List[Action] = [
            monomial for monomial in monomials if not isinstance(monomial, FermionAction)
        ]
        self.fermion_monomials: List[FermionAction] = [
            monomial for monomial in monomials if isinstance(monomial, FermionAction)
        ]
        self.integrator = integrator
        self.hmc_inner = hmc_inner
        if len(self.fermion_monomials) == 0:
            self.is_staggered = False
        else:
            self.is_staggered = isinstance(self.fermion_monomials[0], StaggeredFermionAction)
            for monomial in self.fermion_monomials[1:]:
                if self.is_staggered != isinstance(monomial, StaggeredFermionAction):
                    getLogger().critical(
                        "FermionAction and StaggeredFermionAction cannot be used at the same time", ValueError
                    )
        if not self.is_staggered:
            self._dirac = WilsonDirac(latt_info, 0, 0, 0)
        else:
            self._dirac = StaggeredDirac(latt_info, 0, 0, 0)
        self.gauge_param = self._dirac.gauge_param
        self.obs_param = QudaGaugeObservableParam()
        self.obs_param.remove_staggered_phase = QudaBoolean(self.is_staggered)
        self.fuseFermionAction()
        self.gauge = LatticeGauge(latt_info)
        self.smeared = LatticeGauge(latt_info)
        self.mom = LatticeMom(latt_info)
        self.force = LatticeMom(latt_info)
        self.force_v2 = LatticeGauge(latt_info)

    def fuseFermionAction(self):
        if self.is_staggered:
            from .action.hisq import HISQAction, MultiHISQAction

            hisq_monomials = [monomial for monomial in self.fermion_monomials if isinstance(monomial, HISQAction)]
            if hisq_monomials != []:
                hisq_monomials = [MultiHISQAction(self.latt_info, hisq_monomials)]
            self.fermion_monomials = hisq_monomials + [
                monomial for monomial in self.fermion_monomials if not isinstance(monomial, HISQAction)
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

    def initialize(self, seed: int, gauge: LatticeGauge, mom: Optional[LatticeMom] = None):
        self.initializeRNG(seed)
        self.loadGauge(gauge)
        if mom is None:
            self.loadMom(LatticeMom(gauge.latt_info))
        else:
            self.loadMom(mom)

    def setFermionVerbosity(self, verbosity: QudaVerbosity):
        for fermion_monomial in self.fermion_monomials:
            fermion_monomial.setVerbosity(verbosity)

    def _stoutSmear(self):
        import os
        import cupy as cp

        if not hasattr(self, "_stout_smear"):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stout_smear.cu")) as f:
                self._stout_smear = cp.RawModule(
                    code=f.read(), options=("--std=c++11",), name_expressions=("stout_smear<double, 3>",)
                ).get_function("stout_smear<double, 3>")

        Lx, Ly, Lz, Lt = self.smeared.latt_info.size
        gauge_lexico = self.smeared.lexico(False)
        smeared_lexico = cp.zeros_like(gauge_lexico)
        self._stout_smear((Lx * Ly * Lz, 3, 1), (Lt, 1, 1), (smeared_lexico, gauge_lexico, 1e-15, Lx, Ly, Lz, Lt))
        self.smeared.data[:3] = self.smeared.latt_info.evenodd(smeared_lexico[:3], True, "cupy")

    def _stoutSmearReverse(self):
        import os
        import cupy as cp

        if not hasattr(self, "_compute_lambda_kernel"):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stout_smear.cu")) as f:
                self._compute_lambda_kernel = cp.RawModule(
                    code=f.read(), options=("--std=c++11",), name_expressions=("compute_lambda_kernel<double, 3>",)
                ).get_function("compute_lambda_kernel<double, 3>")
        if not hasattr(self, "_stout_smear_reverse"):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stout_smear.cu")) as f:
                self._stout_smear_reverse = cp.RawModule(
                    code=f.read(), options=("--std=c++11",), name_expressions=("stout_smear_reverse<double, 3>",)
                ).get_function("stout_smear_reverse<double, 3>")

        Lx, Ly, Lz, Lt = self.smeared.latt_info.size
        gauge_lexico = self.smeared.lexico(False)
        force_lexico = self.force_v2.lexico(False)
        lambda_lexico = cp.zeros_like(gauge_lexico)
        self._compute_lambda_kernel(
            (Lx * Ly * Lz, 3, 1), (Lt, 1, 1), (force_lexico, lambda_lexico, gauge_lexico, 1e-15, Lx, Ly, Lz, Lt)
        )
        self._stout_smear_reverse(
            (Lx * Ly * Lz, 3, 1), (Lt, 1, 1), (force_lexico, lambda_lexico, gauge_lexico, 1e-15, Lx, Ly, Lz, Lt)
        )
        print(self.force_v2.data[0, 1, 0, 0, 0, 0])
        print(force_lexico[0, 0, 0, 0, 1])
        print(cp.where(self.force_v2.lexico(False)[:3] - force_lexico[:3] > 1e-12))
        self.force_v2.data[:3] = self.smeared.latt_info.evenodd(force_lexico[:3], True, "cupy")

    def loadGaugeMomSmeared(self):
        if self.gauge is not None and self.smeared is not None and self.force is not None and self.mom is not None:
            saveGaugeQuda(self.gauge.data_ptrs, self.gauge_param)
            self.gauge_param.make_resident_mom = 0
            self.gauge_param.return_result_mom = 1
            momResidentQuda(self.mom.data_ptrs, self.gauge_param)
            self.gauge_param.make_resident_mom = 1
            self.gauge_param.return_result_mom = 0

            self.smeared.data = self.gauge.data
            if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
                self.smeared.setAntiPeriodicT()
            if self.is_staggered:
                self.smeared.staggeredPhase(True)
            self._stoutSmear()
            if self.is_staggered:
                self.smeared.staggeredPhase(False)
            if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
                self.smeared.setAntiPeriodicT()
            self.gauge_param.use_resident_gauge = 0
            loadGaugeQuda(self.smeared.data_ptrs, self.gauge_param)
            self.gauge_param.use_resident_gauge = 1
            self.force.data = 0
            self.force_v2.data = 0

    def loadGaugeMom(self):
        if self.gauge is not None and self.smeared is not None and self.force is not None and self.mom is not None:
            self.smeared.data = self.gauge.data
            if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
                self.smeared.setAntiPeriodicT()
            if self.is_staggered:
                self.smeared.staggeredPhase(True)
            self._stoutSmearReverse()

            import cupy as cp

            uf = cp.einsum("...ab,...bc->...ac", self.gauge.data, self.force_v2.data)
            uf_ah = (uf - uf.conj().swapaxes(axis1=-2, axis2=-1)) / 2
            uf_tlah = uf_ah - (uf_ah.trace(axis1=-2, axis2=-1) / 3)[..., None, None] * cp.eye(3)
            self.force.data[..., 0] = uf_tlah[..., 0, 1].real
            self.force.data[..., 1] = uf_tlah[..., 0, 1].imag
            self.force.data[..., 2] = uf_tlah[..., 0, 2].real
            self.force.data[..., 3] = uf_tlah[..., 0, 2].imag
            self.force.data[..., 4] = uf_tlah[..., 1, 2].real
            self.force.data[..., 5] = uf_tlah[..., 1, 2].imag
            self.force.data[..., 6] = uf_tlah[..., 0, 0].imag
            self.force.data[..., 7] = uf_tlah[..., 1, 1].imag
            self.force.data[..., 8] = uf_tlah[..., 2, 2].imag

            self.mom += self.force

            self.gauge_param.use_resident_gauge = 0
            loadGaugeQuda(self.gauge.data_ptrs, self.gauge_param)
            self.gauge_param.use_resident_gauge = 1
            momResidentQuda(self.mom.data_ptrs, self.gauge_param)

    def seed(self, state):
        seed = self.random.randrange(2**32)
        backend = getArrayBackend()
        if state is None:
            state = arrayRandomGetState(backend)
            arrayRandomSeed(seed, backend)
        else:
            arrayRandomSetState(state, backend)
        return state

    def samplePhi(self):
        R"""
        \phi\sim e^{-S_f}=e^{-\phi^\dagger\mathcal{M}^{-1}\phi}

        \mathcal{M}^{-\frac{1}{2}}\phi=\eta\sim e^{-\eta^\dagger\eta}
        """
        state = self.seed(None)

        self.loadGaugeMomSmeared()
        for monomial in self.fermion_monomials:
            monomial.sample()
        if self.hmc_inner is not None:
            self.hmc_inner.samplePhi()
        self.loadGaugeMom()

        self.seed(state)

    def momAction(self) -> float:
        return momActionQuda(nullptr, self.gauge_param)

    def gaugeAction(self) -> float:
        action = 0
        for monomial in self.gauge_monomials:
            action += monomial.action()
        if self.hmc_inner is not None:
            action += self.hmc_inner.gaugeAction()
        return action

    def fermionAction(self, use_action_param: bool = False) -> float:
        """
        use_action_param: use rational parameters for fermion action istead of molecular dynamics.
        """
        action = 0
        self.loadGaugeMomSmeared()
        for monomial in self.fermion_monomials:
            action += monomial.action()  # if not use_action_param else monomial.actionFA()
        if self.hmc_inner is not None:
            action += self.hmc_inner.fermionAction()
        self.loadGaugeMom()
        return action

    def gaugeForce(self, dt: float):
        for monomial in self.gauge_monomials:
            monomial.force(dt, None)

    def fermionForce(self, dt: float):
        self.loadGaugeMomSmeared()
        for monomial in self.fermion_monomials:
            # monomial.force(dt, self.force)
            monomial.force_v2(dt, self.force_v2)
        self.force_v2 *= dt

        self.loadGaugeMom()

    def updateGauge(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, True, self.gauge_param)

    def updateMom(self, dt: float, gauge: bool = True, fermion: bool = True):
        if gauge:
            self.gaugeForce(dt)
        if fermion:
            self.fermionForce(dt)

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.is_staggered:
            gauge_in.staggeredPhase(False)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeriodicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeriodicT()
        if self.is_staggered:
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
        gaussMomQuda(self.random.randrange(2**32), 1.0)

    def projectSU3(self, tol: float):
        gauge = LatticeGauge(self.latt_info)
        self.saveGauge(gauge)
        gauge.projectSU3(tol)
        self.loadGauge(gauge)

    def integrate(self, t: float, project_tol: float = 0.0):
        self.integrator.integrate(
            self.updateGauge if self.hmc_inner is None else self.hmc_inner.integrate, self.updateMom, t
        )
        if project_tol > 0.0:
            self.projectSU3(project_tol)

    def accept(self, delta_s: float):
        return self.latt_info.mpi_comm.bcast(self.random.random() < exp(min(-delta_s, 0)))

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
        qcharge_density = LatticeReal(self.latt_info)
        self.obs_param.qcharge_density = qcharge_density.data_void_ptr
        self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        return qcharge_density
