from typing import Literal

from ..pyquda import (
    QudaGaugeParam,
    QudaGaugeSmearParam,
    QudaGaugeObservableParam,
    gaussGaugeQuda,
    loadGaugeQuda,
    saveGaugeQuda,
    performGaugeSmearQuda,
    gaugeObservablesQuda,
    projectSU3Quda,
    computeGaugeFixingOVRQuda,
    computeGaugeFixingFFTQuda,
)
from ..field import LatticeInfo, LatticeGauge
from ..enum_quda import QudaBoolean, QudaGaugeSmearType, QudaLinkType, QudaReconstructType

from . import general


class PureGauge:
    latt_info: LatticeInfo
    gauge_param: QudaGaugeParam
    smear_param: QudaGaugeSmearParam
    obs_param: QudaGaugeObservableParam

    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = LatticeInfo(latt_info.global_size)
        link_recon = general.link_recon
        link_recon_sloppy = general.link_recon_sloppy
        # Use QUDA_RECONSTRUCT_NO to ensure slight deviations from SU(3) can be preserved
        recon_no = QudaReconstructType.QUDA_RECONSTRUCT_NO
        if link_recon < recon_no or link_recon_sloppy < recon_no:
            general.link_recon = recon_no
            general.link_recon_sloppy = recon_no
        self.newQudaGaugeParam()
        self.newQudaGaugeSmearParam()
        self.newQudaGaugeObservableParam()
        if link_recon < recon_no or link_recon_sloppy < recon_no:
            general.link_recon = link_recon
            general.link_recon_sloppy = link_recon_sloppy

    def newQudaGaugeParam(self):
        gauge_param = general.newQudaGaugeParam(self.latt_info, 1.0, 0.0)
        self.gauge_param = gauge_param

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

    def saveSmearedGauge(self, gauge: LatticeGauge):
        self.gauge_param.type = QudaLinkType.QUDA_SMEARED_LINKS
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        self.gauge_param.type = QudaLinkType.QUDA_WILSON_LINKS

    def projectSU3(self, gauge: LatticeGauge, tol: float):
        self.gauge_param.use_resident_gauge = 0
        self.gauge_param.return_result_gauge = 1
        projectSU3Quda(gauge.data_ptrs, tol, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1
        self.gauge_param.return_result_gauge = 0

    def smearAPE(self, n_steps: int, alpha: float, dir: int):
        self.smear_param.n_steps = n_steps
        self.smear_param.alpha = alpha
        self.smear_param.meas_interval = n_steps + 1
        if dir == 3:
            self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_APE
        else:
            raise NotImplementedError("Applying APE in 4 dimensions not implemented")
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        performGaugeSmearQuda(self.smear_param, self.obs_param)
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_FALSE

    def smearSTOUT(self, n_steps: int, rho: float, dir: int):
        self.smear_param.n_steps = n_steps
        self.smear_param.rho = rho
        self.smear_param.meas_interval = n_steps + 1
        if dir == 3:
            self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_STOUT
        else:
            self.smear_param.epsilon = 1.0
            self.smear_param.smear_type = QudaGaugeSmearType.QUDA_GAUGE_SMEAR_OVRIMP_STOUT
        self.obs_param.compute_qcharge = QudaBoolean.QUDA_BOOLEAN_TRUE
        performGaugeSmearQuda(self.smear_param, self.obs_param)
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
        # self.obs_param.qcharge_density =
        # self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        # performGaugeSmearQuda(self.obs_param)
        # self.obs_param.compute_qcharge_density = QudaBoolean.QUDA_BOOLEAN_TRUE
        raise NotImplementedError("qchargeDensity not implemented. Confusing size of ndarray.")

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
