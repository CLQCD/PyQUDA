from typing import List

from ..pyquda import newMultigridQuda, destroyMultigridQuda
from ..field import LatticeInfo, LatticeGauge, LatticeFermion
from ..enum_quda import QudaDslashType, QudaInverterType, QudaSolveType, QudaPrecision

from . import Dirac, general


class CloverWilson(Dirac):
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        kappa: float,
        tol: float,
        maxiter: int,
        clover_coeff: float = 0.0,
        clover_xi: float = 1.0,
        geo_block_size: List[List[int]] = None,
    ) -> None:
        super().__init__(latt_info)
        cuda_prec_sloppy = general.cuda_prec_sloppy
        single_prec = QudaPrecision.QUDA_SINGLE_PRECISION
        if geo_block_size is not None and cuda_prec_sloppy < single_prec:
            general.cuda_prec_sloppy = single_prec  # Using half with multigrid doesn't work
        self.mg_instance = None
        self.newQudaGaugeParam()
        self.newQudaMultigridParam(geo_block_size, mass, kappa, 1e-1, 12, 5e-6, 1000, 0, 8)
        self.newQudaInvertParam(mass, kappa, tol, maxiter, clover_coeff, clover_xi)
        if geo_block_size is not None and cuda_prec_sloppy < single_prec:
            general.cuda_prec_sloppy = cuda_prec_sloppy

    def newQudaGaugeParam(self):
        gauge_param = general.newQudaGaugeParam(self.latt_info, 1.0, 0.0)
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
                mass, kappa, geo_block_size, coarse_tol, coarse_maxiter, setup_tol, setup_maxiter, nu_pre, nu_post
            )
            mg_inv_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
        else:
            mg_param, mg_inv_param = None, None
        self.mg_param = mg_param
        self.mg_inv_param = mg_inv_param

    def newQudaInvertParam(
        self, mass: float, kappa: float, tol: float, maxiter: int, clover_coeff: float, clover_xi: float
    ):
        invert_param = general.newQudaInvertParam(
            mass, kappa, tol, maxiter, kappa * clover_coeff, clover_xi, self.mg_param
        )
        if self.mg_param is not None:
            invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
            invert_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
            invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
        else:
            invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
        self.invert_param = invert_param

    def loadGauge(self, gauge: LatticeGauge):
        general.loadClover(gauge, self.gauge_param, self.invert_param)
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

    def invert(self, b: LatticeFermion):
        return general.invert(b, self.invert_param)
