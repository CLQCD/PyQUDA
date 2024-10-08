from .. import getLogger
from ..pointer import Pointers
from ..pyquda import computeHISQForceQuda, dslashQuda, saveGaugeQuda
from ..enum_quda import (
    QudaInverterType,
    QudaMassNormalization,
    QudaMatPCType,
    QudaParity,
    QudaSolutionType,
    QudaSolveType,
    QudaVerbosity,
)
from ..field import LatticeGauge, LatticeStaggeredFermion, LatticeInfo
from ..dirac.hisq import HISQ

nullptr = Pointers("void", 0)

from . import rhmc_param
from .abstract import StaggeredFermionAction


class HISQFermion(StaggeredFermionAction):
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: float,
        tol: float,
        maxiter: int,
        naik_epsilon: float = 0.0,
    ) -> None:
        super().__init__(latt_info)
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)

        kappa = 1 / 2

        self.dirac = HISQ(latt_info, mass, kappa, tol, maxiter, naik_epsilon, None)
        self.phi = LatticeStaggeredFermion(latt_info)
        self.gauge_param = self.dirac.gauge_param
        self.invert_param = self.dirac.invert_param
        self.rhmc_param = rhmc_param.hisq[mass]
        self.coeff = self.dirac.forceCoeff(self.rhmc_param.residue_molecular_dynamics)

        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_MASS_NORMALIZATION
        self.invert_param.verbosity = QudaVerbosity.QUDA_SILENT

    def updateFatLong(self, return_v_link: bool):
        u_link = LatticeGauge(self.latt_info)
        saveGaugeQuda(u_link.data_ptrs, self.gauge_param)
        v_link, w_link = self.dirac.computeWLink(u_link, return_v_link)
        fatlink, longlink = self.dirac.computeXLink(w_link)
        self.dirac.loadFatLongGauge(fatlink, longlink)

        return u_link, v_link, w_link

    def action(self, new_gauge: bool) -> float:
        self.updateFatLong(False)
        self.invert_param.compute_action = 1
        self.dirac.invertMultiShiftPC(
            self.phi,
            self.rhmc_param.offset_molecular_dynamics,
            self.rhmc_param.residue_molecular_dynamics,
        )
        self.invert_param.compute_action = 0
        return self.invert_param.action[0]

    def action2(self) -> float:
        x = self.dirac.invertMultiShiftPC(
            self.phi,
            self.rhmc_param.offset_fermion_action,
            self.rhmc_param.residue_fermion_action,
            self.rhmc_param.norm_fermion_action,
        )
        return x.even.norm2()  # - norm_molecular_dynamics * self.phi.even.norm2()

    def force(self, dt, new_gauge: bool):
        u_link, v_link, w_link = self.updateFatLong(True)
        num = len(self.rhmc_param.offset_molecular_dynamics)
        num_naik = 0 if self.dirac.naik_epsilon == 0.0 else num
        xx = self.dirac.invertMultiShiftPC(
            self.phi,
            self.rhmc_param.offset_molecular_dynamics,
            self.rhmc_param.residue_molecular_dynamics,
        )
        for i in range(num):
            dslashQuda(xx[i].odd_ptr, xx[i].even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        computeHISQForceQuda(
            nullptr,
            dt,
            self.dirac.path_coeff_2,
            self.dirac.path_coeff_1,
            w_link.data_ptrs,
            v_link.data_ptrs,
            u_link.data_ptrs,
            xx.data_ptrs,
            num,
            num_naik,
            self.coeff,
            self.gauge_param,
        )

    def sample(self, noise: LatticeStaggeredFermion, new_gauge: bool):
        self.updateFatLong(False)
        x = self.dirac.invertMultiShiftPC(
            noise,
            self.rhmc_param.offset_pseudo_fermion,
            self.rhmc_param.residue_pseudo_fermion,
            self.rhmc_param.norm_pseudo_fermion,
        )
        self.phi.even = x.even
        self.phi.odd = noise.even
