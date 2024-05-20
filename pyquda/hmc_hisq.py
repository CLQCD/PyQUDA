import numpy

from .pointer import Pointers, ndarrayPointer
from .pyquda import (
    QudaGaugeParam,
    QudaInvertParam,
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
    computeKSLinkQuda,
    computeHISQForceQuda,
    computeGaugeForceQuda,
    computeGaugeLoopTraceQuda,
)
from .enum_quda import (
    QudaLinkType,
    QudaMatPCType,
    QudaSolutionType,
    QudaSolveType,
    QudaStaggeredPhase,
    QudaVerbosity,
    QudaTboundary,
    QudaReconstructType,
    QudaDagType,
)
from .field import Nc, LatticeInfo, LatticeGauge, LatticeStaggeredFermion
from .core import getHISQ

nullptr = Pointers("void", 0)

import numpy as np
import cupy as cp


class HMC:
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        tol: float,
        maxiter: int,
        tadpole_coeff: float = 1.0,
        naik_epsilon: float = 0.0,
    ) -> None:
        assert latt_info.anisotropy == 1.0
        self.dirac = getHISQ(latt_info, mass, tol, maxiter, tadpole_coeff, naik_epsilon)

        self.latt_info = latt_info
        self.updated_fat_long = False
        self.gauge_param: QudaGaugeParam = self.dirac.gauge_param
        self.invert_param: QudaInvertParam = self.dirac.invert_param

        self.gauge_param.staggered_phase_type = QudaStaggeredPhase.QUDA_STAGGERED_PHASE_MILC
        self.gauge_param.staggered_phase_applied = 1
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_ODD_ODD
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE  # This is set to compute action
        self.invert_param.verbosity = QudaVerbosity.QUDA_SILENT

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeriodicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1
        self.updated_fat_long = False

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeriodicT()

    def loadMom(self, mom: LatticeGauge):
        momResidentQuda(mom.data_ptrs, self.gauge_param)

    def saveMom(self, mom: LatticeGauge):
        self.gauge_param.make_resident_mom = 0
        self.gauge_param.return_result_mom = 1
        momResidentQuda(mom.data_ptrs, self.gauge_param)
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0
        momResidentQuda(mom.data_ptrs, self.gauge_param)

    def updateGaugeField(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, True, self.gauge_param)
        loadGaugeQuda(nullptr, self.gauge_param)
        self.updated_fat_long = False

    def computeUVW(self):
        gauge = LatticeGauge(self.latt_info)
        self.saveGauge(gauge)

        ulink = gauge.copy()
        vlink = LatticeGauge(gauge.latt_info)
        wlink = LatticeGauge(gauge.latt_info)

        ulink.staggeredPhase()
        self.gauge_param.staggered_phase_applied = 1

        computeKSLinkQuda(
            vlink.data_ptrs,
            nullptr,
            wlink.data_ptrs,
            ulink.data_ptrs,
            self.dirac.fat7_coeff,
            self.gauge_param,
        )

        return ulink, vlink, wlink

    def computeFermionForce(self, dt, x: LatticeStaggeredFermion):
        self.updateFatLong()
        invertQuda(x.even_ptr, x.odd_ptr, self.invert_param)
        u, v, w = self.computeUVW()
        computeHISQForceQuda(
            nullptr,
            dt,
            self.dirac.level2_coeff,
            self.dirac.fat7_coeff,
            w.data_ptrs,
            v.data_ptrs,
            u.data_ptrs,
            ndarrayPointer(x.even.reshape(1, -1), True),
            1,
            0,
            numpy.array([[1, -1 / 24]], "<f8"),
            self.gauge_param,
        )
        self.gauge_param.staggered_phase_applied = 0

    def computeGaugeForce(self, dt, force, lengths, coeffs, num_paths, max_length):
        computeGaugeForceQuda(
            nullptr,
            nullptr,
            force,
            lengths,
            coeffs,
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

    def gaussMom(self, seed: int):
        gaussMomQuda(seed, 1.0)

    def actionMom(self) -> float:
        return momActionQuda(nullptr, self.gauge_param)

    def actionGauge(self, path, lengths, coeffs, num_paths, max_length) -> float:
        traces = numpy.zeros((num_paths), "<c16")
        computeGaugeLoopTraceQuda(
            traces,
            path,
            lengths,
            coeffs,
            num_paths,
            max_length,
            1,
        )
        return traces.real.sum()

    def actionFermion(self, x: LatticeStaggeredFermion) -> float:
        self.updateFatLong()
        self.invert_param.compute_action = 1
        invertQuda(x.even_ptr, x.odd_ptr, self.invert_param)
        self.invert_param.compute_action = 0
        return self.invert_param.action[0] - self.latt_info.volume_cb2 * Nc

    def plaquette(self):
        return plaqQuda()[0]

    def updateFatLong(self):
        if not self.updated_fat_long:
            gauge = LatticeGauge(self.latt_info)
            self.saveGauge(gauge)
            fatlink, longlink = self.dirac.computeFatLong(gauge)
            self.gauge_param.use_resident_gauge = 0
            self.gauge_param.staggered_phase_applied = 1
            self.gauge_param.type = QudaLinkType.QUDA_ASQTAD_FAT_LINKS
            loadGaugeQuda(fatlink.data_ptrs, self.gauge_param)
            self.gauge_param.type = QudaLinkType.QUDA_ASQTAD_LONG_LINKS
            self.gauge_param.ga_pad = self.gauge_param.ga_pad * 3
            self.gauge_param.staggered_phase_type = QudaStaggeredPhase.QUDA_STAGGERED_PHASE_NO
            loadGaugeQuda(longlink.data_ptrs, self.gauge_param)
            self.gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS
            self.gauge_param.ga_pad = self.gauge_param.ga_pad / 3
            self.gauge_param.staggered_phase_type = QudaStaggeredPhase.QUDA_STAGGERED_PHASE_MILC
            self.gauge_param.staggered_phase_applied = 0
            self.gauge_param.use_resident_gauge = 1
            # self.updated_fat_long = True

    def initNoise(self, x: LatticeStaggeredFermion, seed: int):
        self.updateFatLong()
        self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
        MatQuda(x.odd_ptr, x.even_ptr, self.invert_param)
        self.invert_param.dagger = QudaDagType.QUDA_DAG_NO
