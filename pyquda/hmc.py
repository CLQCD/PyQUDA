from typing import List, Literal

import numpy

from .pointer import Pointers, ndarrayDataPointer
from .pyquda import (
    QudaGaugeObservableParam,
    QudaGaugeParam,
    QudaInvertParam,
    QudaMultigridParam,
    QudaGaugeSmearParam,
    loadCloverQuda,
    freeCloverQuda,
    loadGaugeQuda,
    performGaugeSmearQuda,
    saveGaugeQuda,
    updateGaugeFieldQuda,
    MatQuda,
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
from .enum_quda import (
    QudaBoolean,
    QudaGaugeSmearType,
    QudaLinkType,
    QudaMatPCType,
    QudaSolutionType,
    QudaVerbosity,
    QudaTboundary,
    QudaReconstructType,
    QudaDagType,
)
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
        stout_nstep: int = 0,
        stout_rho: float = 0.0,
        stout_ndim: Literal[3, 4] = 4,
    ) -> None:
        self.dslash = getDslash(
            latt_size, mass, tol, maxiter, clover_coeff_t=clover_coeff, anti_periodic_t=anti_periodic_t
        )

        Lx, Ly, Lz, Lt = latt_size
        self.volume = Lx * Ly * Lz * Lt
        self.updated_clover = False
        self.gauge_param: QudaGaugeParam = self.dslash.gauge_param
        self.invert_param: QudaInvertParam = self.dslash.invert_param
        self.smear_param = QudaGaugeSmearParam()
        self.obs_param = QudaGaugeObservableParam()

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

        self.smear_param.n_steps = stout_nstep
        self.smear_param.rho = stout_rho
        self.smear_param.meas_interval = stout_nstep + 1
        if stout_ndim == 3:
            self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_STOUT
        elif stout_ndim == 4:
            self.smear_param.epsilon = 1.0
            self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_OVRIMP_STOUT

        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeroidicT()
        use_resident_gauge = self.gauge_param.use_resident_gauge
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = use_resident_gauge
        self.updated_clover = False

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeroidicT()

    def updateGaugeField(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, False, self.gauge_param)
        loadGaugeQuda(nullptr, self.gauge_param)
        self.updated_clover = False

    def computeCloverForce(self, dt, x: LatticeFermion, kappa2, ck):
        self.updateClover()
        invertQuda(x.even_ptr, x.odd_ptr, self.invert_param)
        # Some conventions force the dagger to be YES here
        self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
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
        self.invert_param.dagger = QudaDagType.QUDA_DAG_NO

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

    def reunitGaugeField(self, tol: float):
        gauge = LatticeGauge(self.gauge_param.X, None)
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

    def actionFermion(self, x: LatticeFermion) -> float:
        self.updateClover()
        invertQuda(x.even_ptr, x.odd_ptr, self.invert_param)
        return self.invert_param.action[0] - self.volume / 2 * Ns * Nc - 2 * self.invert_param.trlogA[1]

    def updateClover(self):
        if not self.updated_clover:
            freeCloverQuda()
            loadCloverQuda(nullptr, nullptr, self.invert_param)
            self.updated_clover = True

    def initNoise(self, x: LatticeFermion, seed: int):
        self.updateClover()
        self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
        MatQuda(x.odd_ptr, x.even_ptr, self.invert_param)
        self.invert_param.dagger = QudaDagType.QUDA_DAG_NO

    def smearGauge(self):
        t_boundary = self.gauge_param.t_boundary
        reconstruct = self.gauge_param.reconstruct
        _type = self.gauge_param.type
        gauge = LatticeGauge(self.gauge_param.X, None)
        smeared_gauge = LatticeGauge(self.gauge_param.X, None)
        self.saveGauge(gauge)
        self.gauge_param.t_boundary = QudaTboundary.QUDA_PERIODIC_T
        self.gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
        self.loadGauge(gauge)
        performGaugeSmearQuda(self.smear_param, self.obs_param)
        self.gauge_param.type = QudaLinkType.QUDA_SMEARED_LINKS
        self.saveGauge(smeared_gauge)
        self.gauge_param.type = _type
        self.gauge_param.t_boundary = t_boundary
        self.gauge_param.reconstruct = reconstruct
        self.loadGauge(smeared_gauge)
        return gauge
