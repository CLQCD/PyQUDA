from typing import List, Union

from .pointer import Pointers, ndarrayPointer
from .pyquda import (
    gaussMomQuda,
    loadGaugeQuda,
    momActionQuda,
    momResidentQuda,
    plaqQuda,
    projectSU3Quda,
    saveGaugeQuda,
    updateGaugeFieldQuda,
)
from .enum_quda import QudaReconstructType, QudaTboundary
from .field import LatticeInfo, LatticeGauge, LatticeFermion
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

    def actionGauge(self) -> float:
        retval = 0
        for monomial in self.monomials:
            if isinstance(monomial, FermionAction):
                retval += monomial.action(self.new_gauge)
            elif isinstance(monomial, GaugeAction):
                retval += monomial.action()
        self.new_gauge = False
        return retval

    def updateGauge(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, False, self.gauge_param)
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
        self.new_gauge = False

    def samplePhi(self, noise: LatticeFermion):
        for monomial in self.monomials:
            if isinstance(monomial, FermionAction):
                monomial.sample(noise, self.new_gauge)
        self.new_gauge = False

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeroidicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1
        self.new_gauge = True

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeroidicT()

    def loadMom(self, mom: LatticeGauge):
        momResidentQuda(mom.data_ptrs, self.gauge_param)

    def reunitGauge(self, tol: float):
        gauge = LatticeGauge(self.latt_info, None)
        t_boundary = self.gauge_param.t_boundary
        reconstruct = self.gauge_param.reconstruct
        self.saveGauge(gauge)
        self.gauge_param.t_boundary = QudaTboundary.QUDA_PERIODIC_T
        self.gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
        self.loadGauge(gauge)
        projectSU3Quda(nullptr, tol, self.gauge_param)
        self.saveGauge(gauge)
        self.gauge_param.t_boundary = t_boundary
        self.gauge_param.reconstruct = reconstruct
        self.loadGauge(gauge)

    def gaussMom(self, seed: int):
        gaussMomQuda(seed, 1.0)

    def plaquette(self):
        return plaqQuda()[0]
