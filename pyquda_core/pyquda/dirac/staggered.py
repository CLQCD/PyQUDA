from typing import List, Union

from ..field import LatticeInfo, LatticeGauge
from ..enum_quda import QudaDslashType, QudaInverterType

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
            mg_param, mg_inv_param = general.newQudaMultigridParam(mass, kappa, geo_block_size, True)
            mg_inv_param.dslash_type = QudaDslashType.QUDA_ASQTAD_DSLASH
            self.multigrid = Multigrid(mg_param, mg_inv_param)
        else:
            self.multigrid = Multigrid(None, None)

    def newQudaInvertParam(self, mass: float, kappa: float, tol: float, maxiter: int):
        invert_param = general.newQudaInvertParam(mass, kappa, tol, maxiter, 0.0, 1.0, self.multigrid.param)
        invert_param.dslash_type = QudaDslashType.QUDA_STAGGERED_DSLASH
        if self.multigrid.param is None:
            invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param = invert_param

    def loadGauge(self, gauge: LatticeGauge, thin_update_only: bool = False):
        general.loadStaggeredGauge(gauge, self.gauge_param)
        if self.multigrid.instance is None:
            self.newMultigrid()
        else:
            self.updateMultigrid(thin_update_only)
