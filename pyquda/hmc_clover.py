import numpy

from .pointer import Pointers, ndarrayPointer
from .pyquda import (
    QudaGaugeParam,
    QudaInvertParam,
    loadCloverQuda,
    loadGaugeQuda,
    saveGaugeQuda,
    updateGaugeFieldQuda,
    MatQuda,
    invertQuda,
    projectSU3Quda,
    momResidentQuda,
    gaussMomQuda,
    momActionQuda,
    plaqQuda,
    computeCloverForceQuda,
    computeGaugeForceQuda,
    computeGaugeLoopTraceQuda,
)
from .enum_quda import (
    QudaMatPCType,
    QudaSolutionType,
    QudaSolveType,
    QudaVerbosity,
    QudaTboundary,
    QudaReconstructType,
    QudaDagType,
)
from .field import Ns, Nc, LatticeInfo, LatticeGauge, LatticeFermion
from .core import getClover

nullptr = Pointers("void", 0)


class HMC:
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        tol: float,
        maxiter: int,
        clover_coeff: float = 0.0,
    ) -> None:
        assert latt_info.anisotropy == 1.0
        self.dirac = getClover(latt_info, mass, tol, maxiter, 1.0, clover_coeff, 1.0)

        self.latt_info = latt_info
        self.updated_clover = False
        self.gauge_param: QudaGaugeParam = self.dirac.gauge_param
        self.invert_param: QudaInvertParam = self.dirac.invert_param

        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE  # This is set to compute action
        self.invert_param.verbosity = QudaVerbosity.QUDA_SILENT
        self.invert_param.compute_action = 1
        self.invert_param.compute_clover_trlog = 1

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeroidicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1
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
            ndarrayPointer(x.even.reshape(1, -1), True),
            ndarrayPointer(numpy.array([1.0], "<f8")),
            kappa2,
            ck,
            1,
            2,
            self.gauge_param,
            self.invert_param,
        )
        self.invert_param.dagger = QudaDagType.QUDA_DAG_NO

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
        # make_resident_mom = self.gauge_param.make_resident_mom
        # return_result_mom = self.gauge_param.return_result_mom
        # self.gauge_param.make_resident_mom = 1
        # self.gauge_param.return_result_mom = 0
        momResidentQuda(mom.data_ptrs, self.gauge_param)
        # self.gauge_param.make_resident_mom = make_resident_mom
        # self.gauge_param.return_result_mom = return_result_mom

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

    def actionFermion(self, x: LatticeFermion) -> float:
        self.updateClover()
        invertQuda(x.even_ptr, x.odd_ptr, self.invert_param)
        return self.invert_param.action[0] - self.latt_info.volume_cb2 * Ns * Nc - 2 * self.invert_param.trlogA[1]

    def plaquette(self):
        return plaqQuda()[0]

    def updateClover(self):
        if not self.updated_clover:
            # freeCloverQuda()
            loadCloverQuda(nullptr, nullptr, self.invert_param)
            self.updated_clover = True

    def initNoise(self, x: LatticeFermion, seed: int):
        self.updateClover()
        self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
        MatQuda(x.odd_ptr, x.even_ptr, self.invert_param)
        self.invert_param.dagger = QudaDagType.QUDA_DAG_NO
