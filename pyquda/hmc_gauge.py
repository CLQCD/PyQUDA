import numpy

from .pointer import Pointers, ndarrayPointer
from .pyquda import (
    QudaGaugeParam,
    loadGaugeQuda,
    saveGaugeQuda,
    updateGaugeFieldQuda,
    projectSU3Quda,
    momResidentQuda,
    gaussMomQuda,
    momActionQuda,
    plaqQuda,
    computeGaugeForceQuda,
    computeGaugeLoopTraceQuda,
)
from .enum_quda import QudaTboundary, QudaReconstructType
from .field import LatticeInfo, LatticeGauge
from .core import getPureGauge

nullptr = Pointers("void", 0)


class HMC:
    def __init__(
        self,
        latt_info: LatticeInfo,
    ) -> None:
        assert latt_info.anisotropy == 1.0
        self.dirac = getPureGauge(latt_info)

        self.latt_info = latt_info
        self.gauge_param: QudaGaugeParam = self.dirac.gauge_param

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeroidicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeroidicT()

    def updateGaugeField(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, False, self.gauge_param)
        loadGaugeQuda(nullptr, self.gauge_param)

    def computeGaugeForce(self, dt, force, lengths, coeffs, num_paths, max_length):
        computeGaugeForceQuda(
            nullptr,
            nullptr,
            ndarrayPointer(force),
            ndarrayPointer(lengths),
            ndarrayPointer(coeffs),
            num_paths,
            max_length,
            dt,
            self.gauge_param,
        )

    def reunitGaugeField(self, tol: float):
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

    def loadMom(self, mom: LatticeGauge):
        momResidentQuda(mom.data_ptrs, self.gauge_param)

    def gaussMom(self, seed: int):
        gaussMomQuda(seed, 1.0)

    def actionMom(self) -> float:
        return momActionQuda(nullptr, self.gauge_param)

    def actionGauge(self, path, lengths, coeffs, num_paths, max_length) -> float:
        traces = numpy.zeros((num_paths), "<c16")
        computeGaugeLoopTraceQuda(
            ndarrayPointer(traces),
            ndarrayPointer(path),
            ndarrayPointer(lengths),
            ndarrayPointer(coeffs),
            num_paths,
            max_length,
            1,
        )
        return traces.real.sum()

    def plaquette(self):
        return plaqQuda()[0]
