from typing import List, Union

from ..field import LatticeInfo, LatticeGauge
from ..enum_quda import QudaDslashType

from . import general
from .abstract import Multigrid, StaggeredFermionDirac


class HISQDirac(StaggeredFermionDirac):
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        tol: float,
        maxiter: int,
        naik_epsilon: float = 0.0,
        multigrid: Union[List[List[int]], Multigrid, None] = None,
    ) -> None:
        kappa = 1 / 2  # to be compatible with mass normalization
        super().__init__(latt_info)
        self.naik_epsilon = naik_epsilon
        self.newPathCoeff()
        self.newQudaGaugeParam()
        self.newQudaMultigridParam(multigrid, mass, kappa)
        self.newQudaInvertParam(mass, kappa, tol, maxiter)
        self.setPrecision()
        self.setReconstruct()

    def newPathCoeff(self):
        self.path_coeff_1, self.path_coeff_2, self.path_coeff_3 = general.newHISQPathCoeff(1.0)

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
                QudaDslashType.QUDA_ASQTAD_DSLASH, mass, kappa, geo_block_size
            )
            self.multigrid = Multigrid(mg_param, mg_inv_param)
        else:
            self.multigrid = Multigrid(None, None)

    def newQudaInvertParam(self, mass: float, kappa: float, tol: float, maxiter: int):
        invert_param = general.newQudaInvertParam(
            QudaDslashType.QUDA_ASQTAD_DSLASH, mass, kappa, tol, maxiter, 0.0, 1.0, self.multigrid.param
        )
        self.invert_param = invert_param

    def computeULink(self, gauge: LatticeGauge):
        return general.computeULink(gauge, self.gauge_param)

    def computeWLink(self, u_link: LatticeGauge):
        return general.computeWLink(u_link, self.path_coeff_1, self.gauge_param)

    def computeVWLink(self, u_link: LatticeGauge):
        return general.computeVWLink(u_link, self.path_coeff_1, self.gauge_param)

    def computeXLink(self, w_link: LatticeGauge):
        return general.computeXLink(w_link, self.path_coeff_2, self.gauge_param)

    def computeXLinkEpsilon(self, fatlink: LatticeGauge, longlink: LatticeGauge, w_link: LatticeGauge):
        return general.computeXLinkEpsilon(
            fatlink, longlink, w_link, self.path_coeff_3, self.naik_epsilon, self.gauge_param
        )

    def loadFatLongGauge(self, fatlink: LatticeGauge, longlink: LatticeGauge):
        general.loadFatLongGauge(fatlink, longlink, 1.0, self.naik_epsilon, self.gauge_param)

    def loadGauge(self, gauge: LatticeGauge, thin_update_only: bool = False):
        u_link = self.computeULink(gauge)
        w_link = self.computeWLink(u_link)
        fatlink, longlink = self.computeXLink(w_link)
        fatlink, longlink = self.computeXLinkEpsilon(fatlink, longlink, w_link)
        general.loadFatLongGauge(fatlink, longlink, 1.0, self.naik_epsilon, self.gauge_param)
        general.loadMultigrid(self.multigrid, self.invert_param, thin_update_only)

    def freeGauge(self):
        general.freeFatLongGauge()
        general.freeMultigrid(self.multigrid, self.invert_param)
