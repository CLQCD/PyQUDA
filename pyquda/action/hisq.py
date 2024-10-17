from typing import List

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
from ..dirac import HISQDirac

nullptr = Pointers("void", 0)

from .abstract import RationalParam, StaggeredFermionAction


class HISQAction(StaggeredFermionAction):
    dirac: HISQDirac

    def __init__(
        self,
        latt_info: LatticeInfo,
        rational_param: RationalParam,
        tol: float,
        maxiter: int,
        naik_epsilon: float = 0.0,
        verbosity: QudaVerbosity = QudaVerbosity.QUDA_SILENT,
    ) -> None:
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)
        super().__init__(latt_info, HISQDirac(latt_info, 0.0, 1 / 2, tol, maxiter, naik_epsilon, None))

        self.phi = LatticeStaggeredFermion(latt_info)
        self.eta = LatticeStaggeredFermion(latt_info)
        self.rational_param = rational_param
        self.setForceParam()

        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_MASS_NORMALIZATION
        self.invert_param.verbosity = verbosity

    def setForceParam(self):
        coeff = [
            [
                2 * res,
                self.dirac.path_coeff_2[1] * 2 * res,
            ]
            for res in self.rational_param.residue_molecular_dynamics
        ]
        if self.dirac.naik_epsilon != 0.0:
            coeff += [
                [
                    self.dirac.path_coeff_3[0] * self.dirac.naik_epsilon * 2 * res,
                    self.dirac.path_coeff_3[1] * self.dirac.naik_epsilon * 2 * res,
                ]
                for res in self.rational_param.residue_molecular_dynamics
            ]
        self.num = len(self.rational_param.offset_molecular_dynamics)
        self.num_naik = 0 if self.dirac.naik_epsilon == 0.0 else self.num
        self.coeff = numpy.array(coeff, "<f8")

    def updateFatLong(self, return_v_link: bool):
        u_link = LatticeGauge(self.latt_info)
        saveGaugeQuda(u_link.data_ptrs, self.gauge_param)
        v_link, w_link = self.dirac.computeWLink(u_link, return_v_link)
        fatlink, longlink = self.dirac.computeXLink(w_link)
        fatlink, longlink = self.dirac.computeXLinkEpsilon(fatlink, longlink, w_link)
        self.dirac.loadFatLongGauge(fatlink, longlink)

        return u_link, v_link, w_link

    def sample(self, new_gauge: bool):
        self.sampleEta()
        self.updateFatLong(False)
        self.invertMultiShift("pseudo_fermion")

    def action(self, new_gauge: bool) -> float:
        self.updateFatLong(False)
        self.invert_param.compute_action = 1
        self.invertMultiShift("molecular_dynamics")
        self.invert_param.compute_action = 0
        return self.invert_param.action[0]

    def action2(self, new_gauge: bool) -> float:
        self.updateFatLong(False)
        self.invertMultiShift("fermion_action")
        return self.eta.even.norm2()  # - norm_molecular_dynamics * self.phi.even.norm2()

    def force(self, dt, new_gauge: bool):
        u_link, v_link, w_link = self.updateFatLong(True)
        xx = self.invertMultiShift("molecular_dynamics")
        for i in range(self.num):
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
            self.num,
            self.num_naik,
            self.coeff,
            self.gauge_param,
        )


class MultiHISQAction(StaggeredFermionAction):
    dirac: HISQDirac

    def __init__(self, latt_info: LatticeInfo, pseudo_fermions: List[HISQAction]) -> None:
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)
        super().__init__(latt_info, pseudo_fermions[0].dirac)

        self.pseudo_fermions = pseudo_fermions
        self.setForceParam()

    def setForceParam(self):
        num_total = 0
        num_naik_total = 0
        for pseudo_fermion in self.pseudo_fermions:
            num_total += pseudo_fermion.num
            num_naik_total += pseudo_fermion.num_naik
        coeff_all = numpy.zeros((num_total + num_naik_total, 2), "<f8")
        num_current = 0
        num_naik_current = num_total
        for pseudo_fermion in self.pseudo_fermions:
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

    def updateFatLong(self, pseudo_fermion: HISQAction):
        fatlink, longlink = pseudo_fermion.dirac.computeXLinkEpsilon(self.fatlink, self.longlink, self.w_link)
        pseudo_fermion.dirac.loadFatLongGauge(fatlink, longlink)

    def sample(self, new_gauge: bool):
        self.prepareFatLong(False)
        for pseudo_fermion in self.pseudo_fermions:
            pseudo_fermion.sampleEta()
            self.updateFatLong(pseudo_fermion)
            pseudo_fermion.invertMultiShift("pseudo_fermion")

    def action(self, new_gauge: bool) -> float:
        action = 0
        self.prepareFatLong(False)
        for pseudo_fermion in self.pseudo_fermions:
            self.updateFatLong(pseudo_fermion)
            pseudo_fermion.invert_param.compute_action = 1
            pseudo_fermion.invertMultiShift("molecular_dynamics")
            pseudo_fermion.invert_param.compute_action = 0
            action += pseudo_fermion.invert_param.action[0]
        return action

    def action2(self, new_gauge: bool) -> float:
        action = 0
        self.prepareFatLong(False)
        for pseudo_fermion in self.pseudo_fermions:
            self.updateFatLong(pseudo_fermion)
            pseudo_fermion.invertMultiShift("fermion_action")
            action += self.eta.even.norm2()
        return action

    def force(self, dt, new_gauge: bool):
        u_link, v_link, w_link = self.prepareFatLong(True)
        num_current = 0
        xx_all = MultiLatticeStaggeredFermion(self.latt_info, self.num)
        for pseudo_fermion in self.pseudo_fermions:
            self.updateFatLong(pseudo_fermion)
            xx = pseudo_fermion.invertMultiShift("molecular_dynamics")
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
