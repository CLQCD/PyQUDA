from typing import List, Union

from ..field import LatticeInfo, LatticeGauge
from ..enum_quda import QudaDslashType, QudaInverterType, QudaReconstructType, QudaPrecision

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
        multigrid: Union[List[List[int]], Multigrid] = None,
    ) -> None:
        kappa = 1 / 2  # to be compatible with mass normalization
        super().__init__(latt_info)
        self.naik_epsilon = naik_epsilon
        self.newPathCoeff()
        self.newQudaGaugeParam(naik_epsilon)
        self.newQudaMultigridParam(multigrid, mass, kappa)
        self.newQudaInvertParam(mass, kappa, tol, maxiter)
        # Using half with multigrid doesn't work
        if multigrid is not None:
            self.setPrecision(sloppy=max(self.precision.sloppy, QudaPrecision.QUDA_SINGLE_PRECISION))
        else:
            self.setPrecision()
        self.setReconstruct(
            cuda=QudaReconstructType.QUDA_RECONSTRUCT_NO,
            sloppy=QudaReconstructType.QUDA_RECONSTRUCT_NO,
            precondition=QudaReconstructType.QUDA_RECONSTRUCT_NO,
            eigensolver=QudaReconstructType.QUDA_RECONSTRUCT_NO,
        )

    def newPathCoeff(self):
        self.path_coeff_1, self.path_coeff_2, self.path_coeff_3 = general.newPathCoeff(1.0)

    def newQudaGaugeParam(self, naik_epsilon: float):
        gauge_param = general.newQudaGaugeParam(self.latt_info, 1.0, naik_epsilon)
        gauge_param.staggered_phase_applied = 1
        self.gauge_param = gauge_param

    def newQudaMultigridParam(self, multigrid: Union[List[List[int]], Multigrid], mass: float, kappa: float):
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
        invert_param.dslash_type = QudaDslashType.QUDA_ASQTAD_DSLASH
        if self.multigrid.param is None:
            invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param = invert_param

    def computeULink(self, gauge: LatticeGauge):
        return general.computeULink(gauge, self.gauge_param)

    def computeWLink(self, u_link: LatticeGauge, return_v_link: bool):
        return general.computeWLink(u_link, return_v_link, self.path_coeff_1, self.gauge_param)

    def computeXLink(self, w_link: LatticeGauge):
        return general.computeXLink(w_link, self.path_coeff_2, self.gauge_param)

    def computeXLinkEpsilon(self, fatlink: LatticeGauge, longlink: LatticeGauge, w_link: LatticeGauge):
        return general.computeXLinkEpsilon(
            fatlink, longlink, w_link, self.path_coeff_3, self.naik_epsilon, self.gauge_param
        )

    def loadFatLongGauge(self, fatlink: LatticeGauge, longlink: LatticeGauge):
        general.loadFatLongGauge(fatlink, longlink, self.gauge_param)

    def loadGauge(self, gauge: LatticeGauge, thin_update_only: bool = False):
        u_link = self.computeULink(gauge)
        v_link, w_link = self.computeWLink(u_link, False)
        fatlink, longlink = self.computeXLink(w_link)
        fatlink, longlink = self.computeXLinkEpsilon(fatlink, longlink, w_link)
        self.loadFatLongGauge(fatlink, longlink)
        if self.multigrid.instance is None:
            self.newMultigrid()
        else:
            self.updateMultigrid(thin_update_only)

    def destroy(self):
        self.destroyMultigrid()
