from typing import List, Union

from . import getCUDABackend
from .pointer import Pointers
from .pyquda import (
    gaussGaugeQuda,
    gaussMomQuda,
    loadGaugeQuda,
    momActionQuda,
    momResidentQuda,
    plaqQuda,
    saveGaugeQuda,
    setVerbosityQuda,
    updateGaugeFieldQuda,
)
from .enum_quda import QudaTboundary, QudaVerbosity
from .field import Nc, Ns, LatticeInfo, LatticeGauge, LatticeFermion
from .dirac.wilson import Wilson
from .action import FermionAction, GaugeAction

nullptr = Pointers("void", 0)


class HMC:
    def __init__(self, latt_info: LatticeInfo, monomials: List[Union[GaugeAction, FermionAction]]) -> None:
        self.latt_info = latt_info
        self.monomials = monomials
        self.dirac = Wilson(latt_info, 0, 0.125, 0, 0, None)
        self.gauge_param = self.dirac.gauge_param
        self.new_gauge = True

    def setVerbosity(self, verbosity: QudaVerbosity):
        setVerbosityQuda(verbosity, b"\0")

    def actionGauge(self) -> float:
        retval = 0
        for monomial in self.monomials:
            if isinstance(monomial, FermionAction):
                retval += monomial.action(self.new_gauge)
            elif isinstance(monomial, GaugeAction):
                retval += monomial.action()
        # self.new_gauge = False
        return retval

    def updateGauge(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, True, self.gauge_param)
        loadGaugeQuda(nullptr, self.gauge_param)
        self.new_gauge = True

    def actionMom(self) -> float:
        return momActionQuda(nullptr, self.gauge_param)

    def updateMom(self, dt):
        for monomial in self.monomials:
            if isinstance(monomial, FermionAction):
                monomial.force(dt, self.new_gauge)
            elif isinstance(monomial, GaugeAction):
                monomial.force(dt)
        # self.new_gauge = False

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
        for monomial in self.monomials:
            if isinstance(monomial, FermionAction):
                monomial.sample(LatticeFermion(self.latt_info, _noise(backend, random, float64)), self.new_gauge)
        # self.new_gauge = False

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeriodicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1
        self.new_gauge = True

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeriodicT()

    def gaussGauge(self, seed: int):
        gaussGaugeQuda(seed, 1.0)

    def loadMom(self, mom: LatticeGauge):
        momResidentQuda(mom.data_ptrs, self.gauge_param)

    def saveMom(self, mom: LatticeGauge):
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
