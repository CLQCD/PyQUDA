from typing import List, Union

from ..field import LatticeInfo, LatticeGauge
from ..enum_quda import QudaDslashType

from . import general
from .abstract import Multigrid, StaggeredFermionDirac


class StaggeredDirac(StaggeredFermionDirac):
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        tol: float,
        maxiter: int,
        tadpole_coeff: float = 1.0,
        multigrid: Union[List[List[int]], Multigrid, None] = None,
    ) -> None:
        kappa = 1 / 2
        self.tadpole_coeff = tadpole_coeff
        super().__init__(latt_info)
        self.newQudaGaugeParam()
        self.newQudaMultigridParam(multigrid, mass, kappa)
        self.newQudaInvertParam(mass, kappa, tol, maxiter)
        self.setPrecision()
        self.setReconstruct()

    def newQudaGaugeParam(self):
        gauge_param = general.newQudaGaugeParam(self.latt_info)
        gauge_param.staggered_phase_applied = 1
        self.gauge_param = gauge_param

    def newQudaMultigridParam(self, multigrid: Union[List[List[int]], Multigrid, None], mass: float, kappa: float):
        if isinstance(multigrid, Multigrid):
            self.multigrid = multigrid
        elif multigrid is not None:
            geo_block_size = multigrid
            mg_param, mg_inv_param = general.newQudaMultigridParam(
                QudaDslashType.QUDA_STAGGERED_DSLASH, mass, kappa, geo_block_size
            )
            self.multigrid = Multigrid(mg_param, mg_inv_param)
        else:
            self.multigrid = Multigrid(None, None)

    def newQudaInvertParam(self, mass: float, kappa: float, tol: float, maxiter: int):
        invert_param = general.newQudaInvertParam(
            QudaDslashType.QUDA_STAGGERED_DSLASH, mass, kappa, tol, maxiter, 0.0, 1.0, self.multigrid.param
        )
        self.invert_param = invert_param

    def loadGauge(self, gauge: LatticeGauge, thin_update_only: bool = False):
        general.loadStaggeredGauge(gauge, self.gauge_param)
        general.loadMultigrid(self.multigrid, self.invert_param, thin_update_only)

    def freeGauge(self):
        general.freeStaggeredGauge()
        general.freeMultigrid(self.multigrid, self.invert_param)
