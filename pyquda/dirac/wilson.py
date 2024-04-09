from typing import List

from ..pyquda import newMultigridQuda, destroyMultigridQuda
from ..field import LatticeInfo, LatticeGauge
from ..enum_quda import QudaDslashType, QudaInverterType, QudaPrecision

from . import Dirac, general


class Wilson(Dirac):
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        kappa: float,
        tol: float,
        maxiter: int,
        geo_block_size: List[List[int]] = None,
    ) -> None:
        super().__init__(latt_info)
        # Using half with multigrid doesn't work
        if geo_block_size is not None:
            self._setPrecision(sloppy=max(self.precision.sloppy, QudaPrecision.QUDA_SINGLE_PRECISION))
        self.mg_instance = None
        self.newQudaGaugeParam()
        self.newQudaMultigridParam(geo_block_size, mass, kappa, 0.25, 16, 1e-6, 1000, 0, 8)
        self.newQudaInvertParam(mass, kappa, tol, maxiter)

    def newQudaGaugeParam(self):
        gauge_param = general.newQudaGaugeParam(self.latt_info, 1.0, 0.0, self.precision, self.reconstruct)
        self.gauge_param = gauge_param

    def newQudaMultigridParam(
        self,
        geo_block_size: List[List[int]],
        mass: float,
        kappa: float,
        coarse_tol: float,
        coarse_maxiter: int,
        setup_tol: float,
        setup_maxiter: int,
        nu_pre: int,
        nu_post: int,
    ):
        if geo_block_size is not None:
            mg_param, mg_inv_param = general.newQudaMultigridParam(
                mass,
                kappa,
                geo_block_size,
                coarse_tol,
                coarse_maxiter,
                setup_tol,
                setup_maxiter,
                nu_pre,
                nu_post,
                self.precision,
            )
            mg_inv_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
        else:
            mg_param, mg_inv_param = None, None
        self.mg_param = mg_param
        self.mg_inv_param = mg_inv_param

    def newQudaInvertParam(self, mass: float, kappa: float, tol: float, maxiter: int):
        invert_param = general.newQudaInvertParam(mass, kappa, tol, maxiter, 0.0, 1.0, self.mg_param, self.precision)
        invert_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
        if self.mg_param is not None:
            invert_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
        self.invert_param = invert_param

    def loadGauge(self, gauge: LatticeGauge):
        general.loadGauge(gauge, self.gauge_param)
        if self.mg_param is not None:
            if self.mg_instance is not None:
                self.destroy()
            self.mg_instance = newMultigridQuda(self.mg_param)
            self.invert_param.preconditioner = self.mg_instance

    def destroy(self):
        if self.mg_instance is not None:
            destroyMultigridQuda(self.mg_instance)
            self.mg_instance = None
