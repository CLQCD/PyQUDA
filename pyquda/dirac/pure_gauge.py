from typing import Literal

import numpy
from numpy.typing import NDArray

from ..pointer import ndarrayPointer
from ..pyquda import (
    QudaGaugeSmearParam,
    QudaGaugeObservableParam,
    MatQuda,
    computeGaugePathQuda,
    gaussGaugeQuda,
    loadGaugeQuda,
    performWFlowQuda,
    saveGaugeQuda,
    freeUniqueGaugeQuda,
    staggeredPhaseQuda,
    performGaugeSmearQuda,
    gaugeObservablesQuda,
    projectSU3Quda,
    computeGaugeFixingOVRQuda,
    computeGaugeFixingFFTQuda,
)
from ..field import LatticeInfo, LatticeGauge, LatticeFermion, LatticeStaggeredFermion
from ..enum_quda import (
    QudaBoolean,
    QudaDslashType,
    QudaGaugeSmearType,
    QudaLinkType,
    QudaMassNormalization,
    QudaReconstructType,
    QudaSolveType,
)

from . import Gauge, general


class PureGauge(Gauge):
    def __init__(self, latt_info: LatticeInfo) -> None:
        super().__init__(latt_info)
        # Use QUDA_RECONSTRUCT_NO to ensure slight deviations from SU(3) can be preserved
        self._setReconstruct(
            cuda=max(self.reconstruct.cuda, QudaReconstructType.QUDA_RECONSTRUCT_NO),
            sloppy=max(self.reconstruct.sloppy, QudaReconstructType.QUDA_RECONSTRUCT_NO),
            precondition=max(self.reconstruct.precondition, QudaReconstructType.QUDA_RECONSTRUCT_NO),
            eigensolver=max(self.reconstruct.eigensolver, QudaReconstructType.QUDA_RECONSTRUCT_NO),
        )
        self.newQudaGaugeParam()
        self.newQudaInvertParam()
        self.newQudaGaugeSmearParam()
        self.newQudaGaugeObservableParam()

    def newQudaGaugeParam(self):
        gauge_param = general.newQudaGaugeParam(self.latt_info, 1.0, 0.0, self.precision, self.reconstruct)
        self.gauge_param = gauge_param

    def newQudaInvertParam(self):
        invert_param = general.newQudaInvertParam(0, 1 / 8, 0, 0, 0.0, 1.0, None, self.precision)
        invert_param.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE
        invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
        self.invert_param = invert_param

    def newQudaGaugeSmearParam(self):
        smear_param = QudaGaugeSmearParam()
        self.smear_param = smear_param

    def newQudaGaugeObservableParam(self):
        obs_param = QudaGaugeObservableParam()
        self.obs_param = obs_param

    def loadGauge(self, gauge: LatticeGauge):
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)

    def freeGauge(self):
        freeUniqueGaugeQuda(QudaLinkType.QUDA_WILSON_LINKS)

    def saveSmearedGauge(self, gauge: LatticeGauge):
        self.gauge_param.type = QudaLinkType.QUDA_SMEARED_LINKS
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        self.gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS

    def freeSmearedGauge(self):
        freeUniqueGaugeQuda(QudaLinkType.QUDA_SMEARED_LINKS)

    def setCovDev(self):
        self.invert_param.dslash_type = QudaDslashType.QUDA_COVDEV_DSLASH
        self.invert_param.mass = -3
        self.invert_param.kappa = 1 / 2

    def covDev(self, x: LatticeFermion, covdev_mu: int):
        b = LatticeFermion(x.latt_info)
        self.invert_param.covdev_mu = covdev_mu
        MatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def setLaplace(self, laplace3D: int):
        laplaceDim = 3 if laplace3D in [0, 1, 2, 3] else 4
        self.invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
        self.invert_param.mass = laplaceDim - 4
        self.invert_param.kappa = 1 / (2 * laplaceDim)
        self.invert_param.laplace3D = laplace3D

    def laplace(self, x: LatticeStaggeredFermion):
        b = LatticeStaggeredFermion(x.latt_info)
        MatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def staggeredPhase(self, gauge: LatticeGauge):
        self.gauge_param.use_resident_gauge = 0
        self.gauge_param.make_resident_gauge = 0
        self.gauge_param.return_result_gauge = 1
        staggeredPhaseQuda(gauge.data_ptrs, self.gauge_param)
        self.gauge_param.staggered_phase_applied = 1 - self.gauge_param.staggered_phase_applied
        self.gauge_param.use_resident_gauge = 1
        self.gauge_param.make_resident_gauge = 1
        self.gauge_param.return_result_gauge = 0

    def projectSU3(self, gauge: LatticeGauge, tol: float):
        self.gauge_param.use_resident_gauge = 0
        self.gauge_param.make_resident_gauge = 0
        self.gauge_param.return_result_gauge = 1
        projectSU3Quda(gauge.data_ptrs, tol, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1
        self.gauge_param.make_resident_gauge = 1
        self.gauge_param.return_result_gauge = 0

    def path(
        self,
        gauge: LatticeGauge,
        input_path_buf: NDArray[numpy.int32],
        path_length: NDArray[numpy.int32],
        loop_coeff: NDArray[numpy.float64],
    ):
        self.gauge_param.overwrite_gauge = 1
        self.gauge_param.use_resident_gauge = 0
        self.gauge_param.make_resident_gauge = 0
        computeGaugePathQuda(
            gauge.data_ptrs,
            gauge.data_ptrs,
            input_path_buf,
            path_length,
            loop_coeff,
            input_path_buf.shape[1],
            input_path_buf.shape[2],
            1.0,
            self.gauge_param,
        )
        self.gauge_param.overwrite_gauge = 0
        self.gauge_param.use_resident_gauge = 1
        self.gauge_param.make_resident_gauge = 1

    def smearAPE(self, n_steps: int, alpha: float, dir_ignore: int):
        self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_APE
        self.smear_param.n_steps = n_steps
        self.smear_param.alpha = alpha
        self.smear_param.meas_interval = n_steps + 1
        self.smear_param.dir_ignore = dir_ignore
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        performGaugeSmearQuda(self.smear_param, self.obs_param)
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE

    def smearSTOUT(self, n_steps: int, rho: float, dir_ignore: int):
        self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_STOUT
        self.smear_param.n_steps = n_steps
        self.smear_param.rho = rho
        self.smear_param.meas_interval = n_steps + 1
        self.smear_param.dir_ignore = dir_ignore
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        performGaugeSmearQuda(self.smear_param, self.obs_param)
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE

    def smearHYP(self, n_steps: int, alpha1: float, alpha2: float, alpha3: float, dir_ignore: int):
        self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_HYP
        self.smear_param.n_steps = n_steps
        self.smear_param.alpha1 = alpha1
        self.smear_param.alpha2 = alpha2
        self.smear_param.alpha3 = alpha3
        self.smear_param.meas_interval = n_steps + 1
        self.smear_param.dir_ignore = dir_ignore
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        performGaugeSmearQuda(self.smear_param, self.obs_param)
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE

    def flowWilson(self, n_steps: int, epsilon: float, t0: float, restart: bool):
        self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_WILSON_FLOW
        self.smear_param.n_steps = n_steps
        self.smear_param.epsilon = epsilon
        self.smear_param.t0 = t0
        self.smear_param.restart = QudaBoolean.QUDA_BOOLEAN_TRUE if restart else QudaBoolean.QUDA_BOOLEAN_FALSE
        self.smear_param.meas_interval = n_steps + 1
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_TRUE
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        performWFlowQuda(self.smear_param, self.obs_param)
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_FALSE
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE

    def flowSymanzik(self, n_steps: int, epsilon: float, t0: float, restart: bool):
        self.smear_param.n_steps = n_steps
        self.smear_param.epsilon = epsilon
        self.smear_param.t0 = t0
        self.smear_param.restart = QudaBoolean.QUDA_BOOLEAN_TRUE if restart else QudaBoolean.QUDA_BOOLEAN_FALSE
        self.smear_param.meas_interval = n_steps + 1
        self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_SYMANZIK_FLOW
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_TRUE
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        performWFlowQuda(self.smear_param, self.obs_param)
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_FALSE
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE

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
        retval = numpy.zeros((self.latt_info.volume), "<c16")
        self.obs_param.qcharge_density = ndarrayPointer(retval, True)
        self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        return retval

    def gauss(self, seed: int, sigma: float):
        gaussGaugeQuda(seed, sigma)

    def fixingOVR(
        self,
        gauge: LatticeGauge,
        gauge_dir: Literal[3, 4],
        Nsteps: int,
        verbose_interval: int,
        relax_boost: float,
        tolerance: float,
        reunit_interval: int,
        stopWtheta: int,
    ):
        computeGaugeFixingOVRQuda(
            gauge.data_ptrs,
            gauge_dir,
            Nsteps,
            verbose_interval,
            relax_boost,
            tolerance,
            reunit_interval,
            stopWtheta,
            self.gauge_param,
        )

    def fixingFFT(
        self,
        gauge: LatticeGauge,
        gauge_dir: Literal[3, 4],
        Nsteps: int,
        verbose_interval: int,
        alpha: float,
        autotune: int,
        tolerance: float,
        stopWtheta: int,
    ):
        computeGaugeFixingFFTQuda(
            gauge.data_ptrs,
            gauge_dir,
            Nsteps,
            verbose_interval,
            alpha,
            autotune,
            tolerance,
            stopWtheta,
            self.gauge_param,
        )
