from typing import List

from ..pyquda import newMultigridQuda, destroyMultigridQuda
from ..field import LatticeGauge, LatticeFermion
from ..enum_quda import QudaDslashType, QudaInverterType, QudaSolveType, QudaPrecision

from . import abstract
from . import general


class Wilson(abstract.Dslash):
    def __init__(
        self,
        latt_size: List[int],
        kappa: float,
        tol: float,
        maxiter: int,
        xi: float = 1.0,
        t_boundary: int = -1,
        geo_block_size: List[List[int]] = None,
    ) -> None:
        self.mg_instance = None
        self.newQudaGaugeParam(latt_size, xi, t_boundary)
        self.newQudaMultigridParam(geo_block_size, kappa, 1e-1, 12, 5e-6, 1000, 0, 8)
        self.newQudaInvertParam(kappa, tol, maxiter)

    def newQudaGaugeParam(self, latt_size: List[int], anisotropy: float, t_boundary: int):
        gauge_param = general.newQudaGaugeParam(latt_size, anisotropy, t_boundary)
        self.gauge_param = gauge_param

    def newQudaMultigridParam(
        self,
        geo_block_size: List[List[int]],
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
                kappa, geo_block_size, coarse_tol, coarse_maxiter, setup_tol, setup_maxiter, nu_pre, nu_post
            )
            mg_inv_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
        else:
            mg_param, mg_inv_param = None, None
        self.mg_param = mg_param
        self.mg_inv_param = mg_inv_param

    def newQudaInvertParam(self, kappa: float, tol: float, maxiter: int):
        invert_param = general.newQudaInvertParam(kappa, tol, maxiter, 0.0, 1.0, self.mg_param)
        if self.mg_param is not None:
            invert_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
            invert_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
            invert_param.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE
        else:
            invert_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
            invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
            invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE
        self.invert_param = invert_param

    def loadGauge(self, U: LatticeGauge):
        general.loadGauge(U, self.gauge_param)
        if self.mg_param is not None:
            self.gauge_param.cuda_prec_sloppy = QudaPrecision.QUDA_SINGLE_PRECISION
            self.mg_inv_param.cuda_prec_sloppy = QudaPrecision.QUDA_SINGLE_PRECISION
            self.invert_param.cuda_prec_sloppy = QudaPrecision.QUDA_SINGLE_PRECISION
            self.invert_param.cuda_prec_refinement_sloppy = QudaPrecision.QUDA_SINGLE_PRECISION
            if self.mg_instance is not None:
                self.destroy()
            self.mg_instance = newMultigridQuda(self.mg_param)
            self.invert_param.preconditioner = self.mg_instance

    def destroy(self):
        if self.mg_instance is not None:
            destroyMultigridQuda(self.mg_instance)
            self.mg_instance = None

    def invert(self, b: LatticeFermion):
        return general.invert(b, self.invert_param)
