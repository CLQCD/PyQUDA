from typing import List, Literal, Union, overload

import numpy

from pyquda_comm.field import (
    LatticeInfo,
    LatticeGauge,
    LatticeMom,
    LatticeFermion,
    LatticeStaggeredFermion,
    LatticeReal,
)
from ..pyquda import (
    QudaEigParam,
    QudaGaugeSmearParam,
    QudaGaugeObservableParam,
    MatQuda,
    computeGaugePathQuda,
    computeGaugeLoopTraceQuda,
    gaussGaugeQuda,
    gaussMomQuda,
    loadGaugeQuda,
    saveGaugeQuda,
    momResidentQuda,
    performWFlowQuda,
    freeUniqueGaugeQuda,
    staggeredPhaseQuda,
    performGaugeSmearQuda,
    performWuppertalnStep,
    gaugeObservablesQuda,
    projectSU3Quda,
    computeGaugeFixingOVRQuda,
    computeGaugeFixingFFTQuda,
)
from ..enum_quda import (
    QudaBoolean,
    QudaDslashType,
    QudaGaugeSmearType,
    QudaLinkType,
    QudaMassNormalization,
    QudaSolveType,
)

from . import general
from .abstract import Dirac


class GaugeDirac(Dirac):
    def __init__(self, latt_info: LatticeInfo) -> None:
        super().__init__(LatticeInfo(latt_info.global_size))  # Keep periodic t boundary and isotropic
        self.newQudaGaugeParam()
        self.newQudaInvertParam()
        self.newQudaEigParam()
        self.newQudaGaugeSmearParam()
        self.newQudaGaugeObservableParam()
        self.setPrecision()
        self.setReconstruct()

    def newQudaGaugeParam(self):
        gauge_param = general.newQudaGaugeParam(self.latt_info)
        self.gauge_param = gauge_param

    def newQudaInvertParam(self):
        invert_param = general.newQudaInvertParam(QudaDslashType.QUDA_COVDEV_DSLASH, -3, 1 / 2, 0, 0, 0.0, 1.0, None)
        invert_param.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE
        invert_param.mass_normalization = QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
        self.invert_param = invert_param

    def newQudaEigParam(self):
        eig_param = QudaEigParam()
        eig_param.vec_infile = b""
        eig_param.vec_outfile = b""
        eig_param.invert_param = self.invert_param
        self.eig_param = eig_param

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

    def loadMom(self, mom: LatticeMom):
        momResidentQuda(mom.data_ptrs, self.gauge_param)

    def saveFreeMom(self, mom: LatticeMom):
        self.gauge_param.make_resident_mom = 0
        self.gauge_param.return_result_mom = 1
        momResidentQuda(mom.data_ptrs, self.gauge_param)
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0

    def saveSmearedGauge(self, gauge: LatticeGauge):
        self.gauge_param.type = QudaLinkType.QUDA_SMEARED_LINKS
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        self.gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS

    def freeSmearedGauge(self):
        freeUniqueGaugeQuda(QudaLinkType.QUDA_SMEARED_LINKS)

    def covDev(self, x: LatticeFermion, covdev_mu: int):
        b = LatticeFermion(x.latt_info)
        self.invert_param.dslash_type = QudaDslashType.QUDA_COVDEV_DSLASH
        self.invert_param.mass = -3
        self.invert_param.kappa = 1 / 2
        self.invert_param.covdev_mu = covdev_mu
        MatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def laplace(self, x: LatticeStaggeredFermion, laplace3D: int):
        b = LatticeStaggeredFermion(x.latt_info)
        self.invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
        laplaceDim = 3 if laplace3D in [0, 1, 2, 3] else 4
        self.invert_param.mass = laplaceDim - 4
        self.invert_param.kappa = 1 / (2 * laplaceDim)
        self.invert_param.laplace3D = laplace3D
        MatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    @overload
    def wuppertalSmear(self, x: LatticeFermion, n_steps: int, alpha: float) -> LatticeFermion: ...
    @overload
    def wuppertalSmear(self, x: LatticeStaggeredFermion, n_steps: int, alpha: float) -> LatticeStaggeredFermion: ...

    def wuppertalSmear(self, x: Union[LatticeFermion, LatticeStaggeredFermion], n_steps: int, alpha: float):
        if isinstance(x, LatticeStaggeredFermion):
            b = LatticeStaggeredFermion(x.latt_info)
            self.invert_param.dslash_type = QudaDslashType.QUDA_STAGGERED_DSLASH
        else:
            b = LatticeFermion(x.latt_info)
            self.invert_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
        performWuppertalnStep(b.data_ptr, x.data_ptr, self.invert_param, n_steps, alpha)
        return b

    def staggeredPhase(self, gauge: LatticeGauge, applied: bool):
        self.gauge_param.use_resident_gauge = 0
        self.gauge_param.make_resident_gauge = 0
        self.gauge_param.return_result_gauge = 1
        self.gauge_param.staggered_phase_applied = int(applied)
        staggeredPhaseQuda(gauge.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1
        self.gauge_param.make_resident_gauge = 1
        self.gauge_param.return_result_gauge = 0
        self.gauge_param.staggered_phase_applied = int(not applied)

    def projectSU3(self, gauge: LatticeGauge, tol: float):
        self.gauge_param.use_resident_gauge = 0
        self.gauge_param.make_resident_gauge = 0
        self.gauge_param.return_result_gauge = 1
        projectSU3Quda(gauge.data_ptrs, tol, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1
        self.gauge_param.make_resident_gauge = 1
        self.gauge_param.return_result_gauge = 0

    @classmethod
    def _getPath(cls, path: List[int]):
        input_path_buf = numpy.zeros((len(path)), "<i4")
        for j, d in enumerate(path):
            if 0 <= d < 4:
                input_path_buf[j] = d
            elif 4 <= d < 8:
                input_path_buf[j] = 7 - (d - 4)
            else:
                raise ValueError(f"path should be list of int from 0 to 7, but get {path}")
        return input_path_buf, len(path)

    def path(self, gauge: LatticeGauge, paths: List[List[int]]):
        from ..field import LatticeGauge as LatticeGauge_

        gauge_path = LatticeGauge_(gauge.latt_info)
        num_paths = 1
        input_path_buf_x, path_length = GaugeDirac._getPath(paths[0])
        input_path_buf = numpy.zeros((4, 1, path_length + 1), "<i4")
        input_path_buf[0, 0, 0] = 7
        input_path_buf[0, 0, 1:] = input_path_buf_x
        for d in range(1, 4):
            input_path_buf_, path_length_ = GaugeDirac._getPath(paths[d])
            assert path_length_ == path_length, "paths in all directions should have the same shape"
            input_path_buf[d, 0, 0] = 7 - d
            input_path_buf[d, 0, 1:] = input_path_buf_
        max_length = path_length
        path_length = numpy.array([path_length], "<i4")
        loop_coeff = numpy.ones((1), "<f8")
        self.gauge_param.overwrite_gauge = 1
        self.gauge_param.use_resident_gauge = 0
        self.gauge_param.make_resident_gauge = 0
        computeGaugePathQuda(
            gauge_path.data_ptrs,
            gauge.data_ptrs,
            input_path_buf,
            path_length + 1,
            loop_coeff,
            num_paths,
            max_length + 1,
            1.0,
            self.gauge_param,
        )
        self.gauge_param.overwrite_gauge = 0
        self.gauge_param.use_resident_gauge = 1
        self.gauge_param.make_resident_gauge = 1
        return gauge_path

    @classmethod
    def _getLoops(cls, loops: List[List[int]]):
        num_paths = len(loops)
        path_length = numpy.zeros((num_paths), "<i4")
        for i in range(num_paths):
            path_length[i] = len(loops[i])
        max_length = int(numpy.max(path_length))
        input_path_buf = numpy.full((num_paths, max_length), -1, "<i4")
        for i in range(num_paths):
            dx = [0, 0, 0, 0]
            for j, d in enumerate(loops[i]):
                if 0 <= d < 4:
                    dx[d] += 1
                    input_path_buf[i, j] = d
                elif 4 <= d < 8:
                    dx[d - 4] -= 1
                    input_path_buf[i, j] = 7 - (d - 4)
                else:
                    raise ValueError(f"path should be list of int from 0 to 7, but get {loops[i]}")
            if dx != [0, 0, 0, 0]:
                raise ValueError(f"path {loops[i]} is not a loop")
        return input_path_buf, path_length, num_paths, max_length

    def loop(self, gauge: LatticeGauge, loops: List[List[List[int]]], coeff: List[float]):
        from ..field import LatticeGauge as LatticeGauge_

        gauge_loop = LatticeGauge_(gauge.latt_info)
        input_path_buf_x, path_length, num_paths, max_length = GaugeDirac._getLoops(loops[0])
        input_path_buf = numpy.zeros((4, num_paths, max_length + 1), "<i4")
        input_path_buf[0, :, 0] = 7
        input_path_buf[0, :, 1:] = input_path_buf_x
        for d in range(1, 4):
            input_path_buf_, path_length_, num_paths_, max_length_ = GaugeDirac._getLoops(loops[d])
            assert (path_length_ == path_length).all(), "paths in all directions should have the same shape"
            input_path_buf[d, :, 0] = 7 - d
            input_path_buf[d, :, 1:] = input_path_buf_
        loop_coeff = numpy.asarray(coeff, "<f8")
        self.gauge_param.overwrite_gauge = 1
        self.gauge_param.use_resident_gauge = 0
        self.gauge_param.make_resident_gauge = 0
        computeGaugePathQuda(
            gauge_loop.data_ptrs,
            gauge.data_ptrs,
            input_path_buf,
            path_length + 1,
            loop_coeff,
            num_paths,
            max_length + 1,
            1.0,
            self.gauge_param,
        )
        self.gauge_param.overwrite_gauge = 0
        self.gauge_param.use_resident_gauge = 1
        self.gauge_param.make_resident_gauge = 1
        return gauge_loop

    def loopTrace(self, loops: List[List[int]]):
        input_path_buf, path_length, num_paths, max_length = GaugeDirac._getLoops(loops)
        traces = numpy.zeros((num_paths), "<c16")
        loop_coeff = numpy.ones((num_paths), "<f8")
        computeGaugeLoopTraceQuda(
            traces,
            input_path_buf,
            path_length,
            loop_coeff,
            num_paths,
            max_length,
            1.0,
        )
        return traces

    def apeSmear(
        self,
        n_steps: int,
        alpha: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_APE
        self.smear_param.n_steps = n_steps
        self.smear_param.alpha = alpha
        self.smear_param.meas_interval = n_steps + 1
        self.smear_param.dir_ignore = dir_ignore
        self.obs_param.compute_plaquette = QudaBoolean(compute_plaquette)
        self.obs_param.compute_qcharge = QudaBoolean(compute_qcharge)
        performGaugeSmearQuda(self.smear_param, self.obs_param)
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_FALSE
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE

    def stoutSmear(
        self,
        n_steps: int,
        rho: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_STOUT
        self.smear_param.n_steps = n_steps
        self.smear_param.rho = rho
        self.smear_param.meas_interval = n_steps + 1
        self.smear_param.dir_ignore = dir_ignore
        self.obs_param.compute_plaquette = QudaBoolean(compute_plaquette)
        self.obs_param.compute_qcharge = QudaBoolean(compute_qcharge)
        performGaugeSmearQuda(self.smear_param, self.obs_param)
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_FALSE
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE

    def hypSmear(
        self,
        n_steps: int,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_HYP
        self.smear_param.n_steps = n_steps
        self.smear_param.alpha1 = alpha1
        self.smear_param.alpha2 = alpha2
        self.smear_param.alpha3 = alpha3
        self.smear_param.meas_interval = n_steps + 1
        self.smear_param.dir_ignore = dir_ignore
        self.obs_param.compute_plaquette = QudaBoolean(compute_plaquette)
        self.obs_param.compute_qcharge = QudaBoolean(compute_qcharge)
        performGaugeSmearQuda(self.smear_param, self.obs_param)
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_FALSE
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE

    def wilsonFlow(
        self,
        n_steps: int,
        epsilon: float,
        t0: float,
        restart: bool,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_WILSON_FLOW
        self.smear_param.n_steps = n_steps
        self.smear_param.epsilon = epsilon
        self.smear_param.t0 = t0
        self.smear_param.restart = QudaBoolean(restart)
        self.smear_param.meas_interval = n_steps + 1
        self.obs_param.compute_plaquette = QudaBoolean(compute_plaquette)
        self.obs_param.compute_qcharge = QudaBoolean(compute_qcharge)
        performWFlowQuda(self.smear_param, self.obs_param)
        self.obs_param.compute_plaquette = QudaBoolean.QUDA_BOOLEAN_FALSE
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE

    def symanzikFlow(
        self,
        n_steps: int,
        epsilon: float,
        t0: float,
        restart: bool,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_SYMANZIK_FLOW
        self.smear_param.n_steps = n_steps
        self.smear_param.epsilon = epsilon
        self.smear_param.t0 = t0
        self.smear_param.restart = QudaBoolean(restart)
        self.smear_param.meas_interval = n_steps + 1
        self.obs_param.compute_plaquette = QudaBoolean(compute_plaquette)
        self.obs_param.compute_qcharge = QudaBoolean(compute_qcharge)
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
        qcharge_density = LatticeReal(self.latt_info)
        self.obs_param.qcharge_density = qcharge_density.data_void_ptr
        self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        gaugeObservablesQuda(self.obs_param)
        self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        return qcharge_density

    def gaussGauge(self, seed: int, sigma: float):
        gaussGaugeQuda(seed, sigma)

    def gaussMom(self, seed: int, sigma: float):
        gaussMomQuda(seed, sigma)

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
