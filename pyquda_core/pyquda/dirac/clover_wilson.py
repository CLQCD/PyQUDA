from typing import List, Optional, Union

from ..field import LatticeInfo, LatticeGauge, LatticeClover
from ..enum_quda import QudaDslashType

from . import general
from .abstract import Multigrid, FermionDirac


class CloverWilsonDirac(FermionDirac):
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        tol: float,
        maxiter: int,
        clover_csw: float = 0.0,
        clover_xi: float = 1.0,
        multigrid: Union[List[List[int]], Multigrid, None] = None,
    ) -> None:
        kappa = 1 / (2 * (mass + 1 + (latt_info.Nd - 1) / latt_info.anisotropy))
        super().__init__(latt_info)
        self.clover: Optional[LatticeClover] = None
        self.clover_inv: Optional[LatticeClover] = None
        self.newQudaGaugeParam()
        self.newQudaMultigridParam(multigrid, mass, kappa)
        self.newQudaInvertParam(mass, kappa, tol, maxiter, clover_csw, clover_xi)
        self.setPrecision()
        self.setReconstruct()

    def newQudaGaugeParam(self):
        gauge_param = general.newQudaGaugeParam(self.latt_info)
        self.gauge_param = gauge_param

    def newQudaMultigridParam(self, multigrid: Union[List[List[int]], Multigrid, None], mass: float, kappa: float):
        if isinstance(multigrid, Multigrid):
            self.multigrid = multigrid
        elif multigrid is not None:
            geo_block_size = multigrid
            mg_param, mg_inv_param = general.newQudaMultigridParam(
                QudaDslashType.QUDA_CLOVER_WILSON_DSLASH, mass, kappa, geo_block_size
            )
            self.multigrid = Multigrid(mg_param, mg_inv_param)
            self.multigrid.setParam()
        else:
            self.multigrid = Multigrid(None, None)

    def newQudaInvertParam(
        self, mass: float, kappa: float, tol: float, maxiter: int, clover_csw: float, clover_xi: float
    ):
        invert_param = general.newQudaInvertParam(
            QudaDslashType.QUDA_CLOVER_WILSON_DSLASH,
            mass,
            kappa,
            tol,
            maxiter,
            kappa * clover_csw,
            clover_xi,
            self.multigrid.param,
        )
        self.invert_param = invert_param

    def saveClover(self, gauge: LatticeGauge):
        if self.clover is None or self.clover_inv is None:
            self.clover = LatticeClover(gauge.latt_info)
            self.clover_inv = LatticeClover(gauge.latt_info)
        general.saveClover(self.clover, self.clover_inv, gauge, self.gauge_param, self.invert_param)

    def restoreClover(self):
        assert self.clover is not None and self.clover_inv is not None
        general.loadClover(self.clover, self.clover_inv, None, self.gauge_param, self.invert_param)
        general.loadMultigrid(self.multigrid, self.invert_param, True)

    def loadGauge(self, gauge: LatticeGauge, thin_update_only: bool = False):
        general.loadClover(self.clover, self.clover_inv, gauge, self.gauge_param, self.invert_param)
        general.loadGauge(gauge, self.gauge_param)
        general.loadMultigrid(self.multigrid, self.invert_param, thin_update_only)

    def freeGauge(self):
        general.freeClover()
        general.freeGauge()
        general.freeMultigrid(self.multigrid, self.invert_param)
