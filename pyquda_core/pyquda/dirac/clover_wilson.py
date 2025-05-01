from typing import List, Union

from ..field import LatticeInfo, LatticeGauge, LatticeClover
from ..enum_quda import QudaDslashType, QudaPrecision

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
        multigrid: Union[List[List[int]], Multigrid] = None,
    ) -> None:
        kappa = 1 / (2 * (mass + 1 + (latt_info.Nd - 1) / latt_info.anisotropy))
        super().__init__(latt_info)
        self.clover: LatticeClover = None
        self.clover_inv: LatticeClover = None
        self.newQudaGaugeParam()
        self.newQudaMultigridParam(multigrid, mass, kappa)
        self.newQudaInvertParam(mass, kappa, tol, maxiter, clover_csw, clover_xi)
        # Using half with multigrid doesn't work
        if multigrid is not None:
            self.setPrecision(sloppy=max(self.precision.sloppy, QudaPrecision.QUDA_SINGLE_PRECISION))
        else:
            self.setPrecision()
        self.setReconstruct()

    def newQudaGaugeParam(self):
        gauge_param = general.newQudaGaugeParam(self.latt_info, 1.0, 0.0)
        self.gauge_param = gauge_param

    def newQudaMultigridParam(self, multigrid: Union[List[List[int]], Multigrid], mass: float, kappa: float):
        if isinstance(multigrid, Multigrid):
            self.multigrid = multigrid
        elif multigrid is not None:
            geo_block_size = multigrid
            mg_param, mg_inv_param = general.newQudaMultigridParam(mass, kappa, geo_block_size, False)
            mg_inv_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
            self.multigrid = Multigrid(mg_param, mg_inv_param)
            self.multigrid.setParam()
        else:
            self.multigrid = Multigrid(None, None)

    def newQudaInvertParam(
        self, mass: float, kappa: float, tol: float, maxiter: int, clover_csw: float, clover_xi: float
    ):
        invert_param = general.newQudaInvertParam(
            mass, kappa, tol, maxiter, kappa * clover_csw, clover_xi, self.multigrid.param
        )
        invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
        self.invert_param = invert_param

    def saveClover(self, gauge: LatticeGauge):
        if self.clover is None or self.clover_inv is None:
            self.clover = LatticeClover(gauge.latt_info)
            self.clover_inv = LatticeClover(gauge.latt_info)
        general.saveClover(self.clover, self.clover_inv, gauge, self.gauge_param, self.invert_param)

    def restoreClover(self):
        assert self.clover is not None and self.clover_inv is not None
        general.loadClover(self.clover, self.clover_inv, None, self.gauge_param, self.invert_param)
        self.updateMultigrid(True)

    def loadGauge(self, gauge: LatticeGauge, thin_update_only: bool = False):
        general.loadClover(self.clover, self.clover_inv, gauge, self.gauge_param, self.invert_param)
        general.loadGauge(gauge, self.gauge_param)
        if self.multigrid.instance is None:
            self.newMultigrid()
        else:
            self.updateMultigrid(thin_update_only)

    def destroy(self):
        self.destroyMultigrid()
