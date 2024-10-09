from typing import List, Sequence, Union

import numpy

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
from ..field import LatticeInfo, LatticeGauge, LatticeStaggeredFermion, MultiLatticeStaggeredFermion
from ..dirac.hisq import HISQ

nullptr = Pointers("void", 0)

from . import rhmc_param
from .abstract import StaggeredFermionAction


class HISQFermion(StaggeredFermionAction):
    def __init__(
        self,
        latt_info: LatticeInfo,
        mass: Union[float, Sequence[float]],
        flavor: Union[int, Sequence[int]],
        tol: float,
        maxiter: int,
        naik_epsilon: float = 0.0,
    ) -> None:
        super().__init__(latt_info)
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)
        mass = (mass,) if not hasattr(mass, "__iter__") else tuple(mass)
        flavor = (flavor,) if not hasattr(flavor, "__iter__") else tuple(flavor)
        assert len(mass) == len(flavor)

        self.dirac = HISQ(latt_info, 0.0, 1 / 2, tol, maxiter, naik_epsilon, None)
        self.phi = LatticeStaggeredFermion(latt_info)
        self.gauge_param = self.dirac.gauge_param
        self.invert_param = self.dirac.invert_param
        self.rhmc_param = rhmc_param.staggered[(mass, flavor)]

        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_MASS_NORMALIZATION
        self.invert_param.verbosity = QudaVerbosity.QUDA_SILENT

        self.num = len(self.rhmc_param.offset_molecular_dynamics)
        self.num_naik = 0 if naik_epsilon == 0.0 else self.num
        self.coeff = self.dirac.forceCoeff(self.rhmc_param.residue_molecular_dynamics)

    def updateFatLong(self, return_v_link: bool):
        u_link = LatticeGauge(self.latt_info)
        saveGaugeQuda(u_link.data_ptrs, self.gauge_param)
        v_link, w_link = self.dirac.computeWLink(u_link, return_v_link)
        fatlink, longlink = self.dirac.computeXLink(w_link)
        fatlink, longlink = self.dirac.computeXLinkEpsilon(fatlink, longlink, w_link)
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


class MultiHISQFermion(StaggeredFermionAction):
    def __init__(self, latt_info: LatticeInfo, pseudo_fermions: List[HISQFermion]) -> None:
        super().__init__(latt_info)
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)

        self.dirac = HISQ(latt_info, 0.0, 0.5, 0, 0, 0.0, None)
        self.gauge_param = self.dirac.gauge_param
        self.invert_param = self.dirac.invert_param

        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_MASS_NORMALIZATION
        self.invert_param.verbosity = QudaVerbosity.QUDA_SILENT

        self.pseudo_fermions = pseudo_fermions

        num_total = 0
        num_naik_total = 0
        for pseudo_fermion in pseudo_fermions:
            num_total += pseudo_fermion.num
            num_naik_total += pseudo_fermion.num_naik
        coeff_all = numpy.zeros((num_total + num_naik_total, 2), "<f8")
        num_current = 0
        num_naik_current = num_total
        for pseudo_fermion in pseudo_fermions:
            num = pseudo_fermion.num
            num_naik = pseudo_fermion.num_naik
            coeff_all[num_current : num_current + num] = pseudo_fermion.coeff[:num]
            coeff_all[num_naik_current : num_naik_current + num_naik] = pseudo_fermion.coeff[num:]
            num_current += num
            num_naik_current += num_naik
        self.num = num_total
        self.num_naik = num_naik_total
        self.coeff = coeff_all

    def prepareFatLong(self, return_v_link: bool):
        u_link = LatticeGauge(self.latt_info)
        saveGaugeQuda(u_link.data_ptrs, self.gauge_param)
        v_link, w_link = self.dirac.computeWLink(u_link, return_v_link)
        self.fatlink, self.longlink = self.dirac.computeXLink(w_link)
        self.w_link = w_link

        return u_link, v_link, w_link

    def updateFatLong(self, pseudo_fermion: HISQFermion):
        fatlink, longlink = pseudo_fermion.dirac.computeXLinkEpsilon(self.fatlink, self.longlink, self.w_link)
        pseudo_fermion.dirac.loadFatLongGauge(fatlink, longlink)

    def action(self, new_gauge: bool) -> float:
        action = 0
        self.prepareFatLong(False)
        for pseudo_fermion in self.pseudo_fermions:
            self.updateFatLong(pseudo_fermion)
            pseudo_fermion.invert_param.compute_action = 1
            pseudo_fermion.dirac.invertMultiShiftPC(
                pseudo_fermion.phi,
                pseudo_fermion.rhmc_param.offset_molecular_dynamics,
                pseudo_fermion.rhmc_param.residue_molecular_dynamics,
            )
            pseudo_fermion.invert_param.compute_action = 0
            action += pseudo_fermion.invert_param.action[0]
        return action

    def action2(self) -> float:
        action = 0
        self.prepareFatLong(False)
        for pseudo_fermion in self.pseudo_fermions:
            self.updateFatLong(pseudo_fermion)
            x = pseudo_fermion.dirac.invertMultiShiftPC(
                pseudo_fermion.phi,
                pseudo_fermion.rhmc_param.offset_fermion_action,
                pseudo_fermion.rhmc_param.residue_fermion_action,
                pseudo_fermion.rhmc_param.norm_fermion_action,
            )
            action += x.even.norm2()
        return action

    def force(self, dt, new_gauge: bool):
        u_link, v_link, w_link = self.prepareFatLong(True)
        num_current = 0
        xx_all = MultiLatticeStaggeredFermion(self.latt_info, self.num)
        for pseudo_fermion in self.pseudo_fermions:
            self.updateFatLong(pseudo_fermion)
            xx = pseudo_fermion.dirac.invertMultiShiftPC(
                pseudo_fermion.phi,
                pseudo_fermion.rhmc_param.offset_molecular_dynamics,
                pseudo_fermion.rhmc_param.residue_molecular_dynamics,
            )
            for i in range(pseudo_fermion.num):
                dslashQuda(xx[i].odd_ptr, xx[i].even_ptr, pseudo_fermion.invert_param, QudaParity.QUDA_ODD_PARITY)
                xx_all[num_current + i] = xx[i]
            num_current += pseudo_fermion.num
        computeHISQForceQuda(
            nullptr,
            dt,
            self.dirac.path_coeff_2,
            self.dirac.path_coeff_1,
            w_link.data_ptrs,
            v_link.data_ptrs,
            u_link.data_ptrs,
            xx_all.data_ptrs,
            self.num,
            self.num_naik,
            self.coeff,
            self.gauge_param,
        )

    def sample(self, noise: List[LatticeStaggeredFermion], new_gauge: bool):
        self.prepareFatLong(False)
        for idx, pseudo_fermion in enumerate(self.pseudo_fermions):
            self.updateFatLong(pseudo_fermion)
            x = pseudo_fermion.dirac.invertMultiShiftPC(
                noise[idx],
                pseudo_fermion.rhmc_param.offset_pseudo_fermion,
                pseudo_fermion.rhmc_param.residue_pseudo_fermion,
                pseudo_fermion.rhmc_param.norm_pseudo_fermion,
            )
            pseudo_fermion.phi.even = x.even
            pseudo_fermion.phi.odd = noise[idx].even
