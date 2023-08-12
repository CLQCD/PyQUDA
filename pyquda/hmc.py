from typing import List

import numpy

from .pointer import Pointers, ndarrayDataPointer
from .pyquda import (
    QudaGaugeParam,
    QudaInvertParam,
    QudaMultigridParam,
    loadCloverQuda,
    freeCloverQuda,
    loadGaugeQuda,
    saveGaugeQuda,
    updateGaugeFieldQuda,
    invertQuda,
    projectSU3Quda,
    momResidentQuda,
    gaussMomQuda,
    momActionQuda,
    computeCloverForceQuda,
    computeGaugeForceQuda,
    computeGaugeLoopTraceQuda,
)
from .field import Ns, Nc
from .enum_quda import QudaMatPCType, QudaSolutionType, QudaVerbosity, QudaTboundary, QudaReconstructType
from .core import LatticeGauge, LatticeFermion, getDslash

nullptr = Pointers("void", 0)


class HMC:
    def __init__(
        self,
        latt_size: List[int],
        mass: float,
        tol: float,
        maxiter: int,
        clover_coeff: float = 0.0,
        anti_periodic_t=True,
    ) -> None:
        self.dslash = getDslash(
            latt_size, mass, tol, maxiter, clover_coeff_t=clover_coeff, anti_periodic_t=anti_periodic_t
        )
        Lx, Ly, Lz, Lt = latt_size
        self.volume = Lx * Ly * Lz * Lt
        self.gauge_param: QudaGaugeParam = self.dslash.gauge_param
        self.invert_param: QudaInvertParam = self.dslash.invert_param

        self.gauge_param.overwrite_gauge = 0
        self.gauge_param.overwrite_mom = 0
        self.gauge_param.use_resident_gauge = 1
        self.gauge_param.use_resident_mom = 1
        self.gauge_param.make_resident_gauge = 1
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_gauge = 0
        self.gauge_param.return_result_mom = 0

        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
        self.invert_param.verbosity = QudaVerbosity.QUDA_SILENT
        self.invert_param.compute_action = 1
        self.invert_param.compute_clover_trlog = 1

    def loadGauge(self, gauge: LatticeGauge):
        use_resident_gauge = self.gauge_param.use_resident_gauge

        gauge_data_bak = gauge.backup()
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeroidicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = use_resident_gauge
        gauge.data = gauge_data_bak

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)

    def updateGaugeField(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, False, self.gauge_param)

    def computeCloverForce(self, dt, x: LatticeFermion, kappa2, ck):
        self.updateClover()
        invertQuda(x.even_ptr, x.odd_ptr, self.invert_param)
        computeCloverForceQuda(
            nullptr,
            dt,
            ndarrayDataPointer(x.even.reshape(1, -1), True),
            nullptr,
            ndarrayDataPointer(numpy.array([1.0], "<f8")),
            kappa2,
            ck,
            1,
            2,
            nullptr,
            self.gauge_param,
            self.invert_param,
        )

    def computeGaugeForce(self, dt, force, lengths, coeffs, num_paths, max_length):
        computeGaugeForceQuda(
            nullptr,
            nullptr,
            ndarrayDataPointer(force),
            ndarrayDataPointer(lengths),
            ndarrayDataPointer(coeffs),
            num_paths,
            max_length,
            dt,
            self.gauge_param,
        )

    def reunitGaugeField(self, ref: LatticeGauge, tol: float):
        gauge = LatticeGauge(self.gauge_param.X, None, ref.t_boundary)
        t_boundary = self.gauge_param.t_boundary
        anisotropy = self.gauge_param.anisotropy
        reconstruct = self.gauge_param.reconstruct
        use_resident_gauge = self.gauge_param.use_resident_gauge

        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeroidicT()
        if anisotropy != 1.0:
            gauge.setAnisotropy(1 / anisotropy)

        self.gauge_param.t_boundary = QudaTboundary.QUDA_PERIODIC_T
        self.gauge_param.anisotropy = 1.0
        self.gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = use_resident_gauge
        projectSU3Quda(nullptr, tol, self.gauge_param)
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        self.gauge_param.t_boundary = t_boundary
        self.gauge_param.anisotropy = anisotropy
        self.gauge_param.reconstruct = reconstruct

        if t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeroidicT()
        if anisotropy != 1.0:
            gauge.setAnisotropy(anisotropy)
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = use_resident_gauge

    def loadMom(self, mom: LatticeGauge):
        make_resident_mom = self.gauge_param.make_resident_mom
        return_result_mom = self.gauge_param.return_result_mom
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0
        momResidentQuda(mom.data_ptrs, self.gauge_param)
        self.gauge_param.make_resident_mom = make_resident_mom
        self.gauge_param.return_result_mom = return_result_mom

    def gaussMom(self, seed: int):
        gaussMomQuda(seed, 1.0)

    def actionMom(self) -> float:
        return momActionQuda(nullptr, self.gauge_param)

    def actionGauge(self, path, lengths, coeffs, num_paths, max_length) -> float:
        traces = numpy.zeros((num_paths), "<c16")
        computeGaugeLoopTraceQuda(
            ndarrayDataPointer(traces),
            ndarrayDataPointer(path),
            ndarrayDataPointer(lengths),
            ndarrayDataPointer(coeffs),
            num_paths,
            max_length,
            1,
        )
        return traces.real.sum()

    def actionFermion(self) -> float:
        return self.invert_param.action[0] - self.volume / 2 * Ns * Nc - 2 * self.invert_param.trlogA[1]

    def updateClover(self):
        freeCloverQuda()
        loadCloverQuda(nullptr, nullptr, self.invert_param)
