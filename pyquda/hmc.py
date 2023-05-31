from typing import List

import numpy as np

from .dslash.abstract import Dslash
from .pyquda import (
    Pointer, ndarrayDataPointer, QudaGaugeParam, QudaInvertParam, QudaMultigridParam, loadCloverQuda, freeCloverQuda,
    loadGaugeQuda, saveGaugeQuda, updateGaugeFieldQuda, invertQuda, dslashQuda, cloverQuda, projectSU3Quda,
    momResidentQuda, gaussMomQuda, momActionQuda, computeCloverForceQuda, computeGaugeForceQuda,
    computeGaugeLoopTraceQuda
)
from .field import Ns, Nc
from .enum_quda import (QudaMatPCType, QudaSolutionType, QudaVerbosity)
from .core import LatticeGauge, LatticeFermion

nullptr = Pointer("void", 0)


class HMC:
    def __init__(self, latt_size: List[int], dslash: Dslash) -> None:
        Lx, Ly, Lz, Lt = latt_size
        self.volume = Lx * Ly * Lz * Lt
        self.gauge_param: QudaGaugeParam = dslash.gauge_param
        self.invert_param: QudaInvertParam = dslash.invert_param

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
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge.data_ptr, self.gauge_param)
        self.gauge_param.use_resident_gauge = use_resident_gauge

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptr, self.gauge_param)

    def updateGaugeField(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, False, self.gauge_param)

    def computeCloverForce(self, dt, x: LatticeFermion, kappa2, ck):
        self.updateClover()
        invertQuda(x.even_ptr, x.odd_ptr, self.invert_param)
        computeCloverForceQuda(
            nullptr, dt, ndarrayDataPointer(x.even.reshape(1, -1), True), nullptr,
            ndarrayDataPointer(np.array([1.0], "<f8")), kappa2, ck, 1, 2, nullptr, self.gauge_param, self.invert_param
        )

    def computeGaugeForce(self, dt, force, lengths, coeffs, num_paths, max_length):
        computeGaugeForceQuda(
            nullptr, nullptr, ndarrayDataPointer(force), ndarrayDataPointer(lengths), ndarrayDataPointer(coeffs),
            num_paths, max_length, dt, self.gauge_param
        )

    def reunitGaugeField(self, tol: float):
        projectSU3Quda(nullptr, tol, self.gauge_param)

    def loadMom(self, mom: LatticeGauge):
        make_resident_mom = self.gauge_param.make_resident_mom
        return_result_mom = self.gauge_param.return_result_mom
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0
        momResidentQuda(mom.data_ptr, self.gauge_param)
        self.gauge_param.make_resident_mom = make_resident_mom
        self.gauge_param.return_result_mom = return_result_mom

    def gaussMom(self, seed: int):
        gaussMomQuda(seed, 1.0)

    def actionMom(self) -> float:
        return momActionQuda(self.gauge_param)

    def actionGauge(self, path, lengths, coeffs, num_paths, max_length) -> float:
        traces = np.zeros((num_paths), "<c16")
        computeGaugeLoopTraceQuda(
            ndarrayDataPointer(traces), ndarrayDataPointer(path), ndarrayDataPointer(lengths),
            ndarrayDataPointer(coeffs), num_paths, max_length, 1
        )
        return traces.real.sum()

    def actionFermion(self) -> float:
        return self.invert_param.action[0] - self.volume / 2 * Ns * Nc - self.invert_param.trlogA[1]

    def updateClover(self):
        freeCloverQuda()
        loadCloverQuda(nullptr, nullptr, self.invert_param)
