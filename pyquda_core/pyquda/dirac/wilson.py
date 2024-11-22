from typing import List, Union

from ..field import LatticeInfo, LatticeGauge
from ..enum_quda import QudaDslashType, QudaPrecision

from . import general
from .abstract import Multigrid, FermionDirac


class WilsonDirac(FermionDirac):
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        tol: float,
        maxiter: int,
        multigrid: Union[List[List[int]], Multigrid] = None,
    ) -> None:
        kappa = 1 / (2 * (mass + 1 + (latt_info.Nd - 1) / latt_info.anisotropy))
        super().__init__(latt_info)
        self.newQudaGaugeParam()
        self.newQudaMultigridParam(multigrid, mass, kappa, 0.25, 16, 1e-6, 1000, 0, 8)
        self.newQudaInvertParam(mass, kappa, tol, maxiter)
        # Using half with multigrid doesn't work
        if multigrid is not None:
            self.setPrecision(sloppy=max(self.precision.sloppy, QudaPrecision.QUDA_SINGLE_PRECISION))
        else:
            self.setPrecision()
        self.setReconstruct()

    def newQudaGaugeParam(self):
        gauge_param = general.newQudaGaugeParam(self.latt_info, 1.0, 0.0)
        self.gauge_param = gauge_param

    def newQudaMultigridParam(
        self,
        multigrid: Union[List[List[int]], Multigrid],
        mass: float,
        kappa: float,
        coarse_tol: float,
        coarse_maxiter: int,
        setup_tol: float,
        setup_maxiter: int,
        nu_pre: int,
        nu_post: int,
    ):
        if isinstance(multigrid, Multigrid):
            self.multigrid = multigrid
        elif multigrid is not None:
            geo_block_size = multigrid
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
            )
            mg_inv_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
            self.multigrid = Multigrid(mg_param, mg_inv_param)
        else:
            self.multigrid = Multigrid(None, None)

    def newQudaInvertParam(self, mass: float, kappa: float, tol: float, maxiter: int):
        invert_param = general.newQudaInvertParam(mass, kappa, tol, maxiter, 0.0, 1.0, self.multigrid.param)
        invert_param.dslash_type = QudaDslashType.QUDA_WILSON_DSLASH
        self.invert_param = invert_param

    def loadGauge(self, gauge: LatticeGauge, thin_update_only: bool = False):
        general.loadGauge(gauge, self.gauge_param)
        if self.multigrid.instance is None:
            self.newMultigrid()
        else:
            self.updateMultigrid(thin_update_only)

    def destroy(self):
        self.destroyMultigrid()
