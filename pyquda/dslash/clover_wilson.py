from typing import List

from ..pyquda import newMultigridQuda, destroyMultigridQuda

from ..core import LatticeGauge, LatticeFermion
from ..enum_quda import QudaDslashType, QudaInverterType, QudaSolveType

from . import abstract
from . import general


class CloverWilson(abstract.Dslash):
    def __init__(
        self,
        latt_size: List[int],
        kappa: float,
        tol: float,
        maxiter: int,
        xi: float = 1.0,
        clover_coeff: float = 0.0,
        clover_xi: float = 1.0,
        t_boundary: int = -1,
        multigrid: bool = False,
    ) -> None:
        self.newQudaGaugeParam(latt_size, xi, t_boundary)
        self.newQudaMultigridParam(multigrid)
        self.newQudaInvertParam(kappa, tol, maxiter, clover_coeff, clover_xi)

    def newQudaGaugeParam(self, latt_size: List[int], anisotropy: float, t_boundary: int):
        gauge_param = general.newQudaGaugeParam(latt_size, anisotropy, t_boundary)
        self.gauge_param = gauge_param

    def newQudaMultigridParam(self, multigrid: bool):
        if multigrid:
            mg_param = general.newQudaMultigridParam(
                [[4, 4, 4, 4, 1, 1], [2, 2, 2, 2, 1, 1]], 1e-12, 12, 5e-6, 1000, 0, 8
            )
        else:
            mg_param = None
        self.mg_param = mg_param

    def newQudaInvertParam(self, kappa: float, tol: float, maxiter: int, clover_coeff: float, clover_xi: float):
        invert_param = general.newQudaInvertParam(kappa, tol, maxiter, kappa * clover_coeff, clover_xi, self.mg_param)
        if self.mg_param is not None:
            invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
            invert_param.inv_type = QudaInverterType.QUDA_GCR_INVERTER
            invert_param.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE
        else:
            invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
            invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
            invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE
        self.invert_param = invert_param

    def loadGauge(self, U: LatticeGauge):
        general.loadClover(U, self.gauge_param, self.invert_param)
        general.loadGauge(U, self.gauge_param)
        if self.mg_param is not None:
            self.mg_instance = newMultigridQuda(self.mg_param)
            self.invert_param.preconditioner = self.mg_instance

    def destroy(self):
        destroyMultigridQuda(self.mg_instance)

    def invert(self, b: LatticeFermion):
        return general.invert(b, self.invert_param)
