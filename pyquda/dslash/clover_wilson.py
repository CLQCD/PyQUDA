from typing import List

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
    ) -> None:

        self.newQudaGaugeParam(latt_size, xi, t_boundary)
        self.newQudaInvertParam(kappa, tol, maxiter, clover_coeff, clover_xi)

    def newQudaGaugeParam(self, latt_size: List[int], anisotropy: float, t_boundary: int):
        gauge_param = general.newQudaGaugeParam(latt_size, anisotropy, t_boundary)
        self.gauge_param = gauge_param

    def newQudaInvertParam(self, kappa: float, tol: float, maxiter: int, clover_coeff: float, clover_xi: float):
        invert_param = general.newQudaInvertParam(kappa, tol, maxiter, kappa * clover_coeff, clover_xi)
        invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
        invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        invert_param.solve_type = QudaSolveType.QUDA_NORMOP_PC_SOLVE
        self.invert_param = invert_param

    def loadGauge(self, U: LatticeGauge):
        general.loadClover(U, self.gauge_param, self.invert_param)
        general.loadGauge(U, self.gauge_param)

    def invert(self, b: LatticeFermion):
        return general.invert(b, self.invert_param)
