from typing import List, Optional

import numpy

from pyquda_comm import getLogger
from ..field import LatticeInfo, LatticeGauge, LatticeMom, LatticeStaggeredFermion, MultiLatticeStaggeredFermion
from ..quda import computeHISQForceQuda, dslashQuda, saveGaugeQuda
from ..enum_quda import (
    QudaReconstructType,
    QudaInverterType,
    QudaMassNormalization,
    QudaMatPCType,
    QudaParity,
    QudaSolutionType,
    QudaSolveType,
    QudaVerbosity,
)
from ..dirac import HISQDirac, getGlobalReconstruct

nullptr = numpy.empty((0, 0), "<c16")

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
    ) -> None:
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)
        super().__init__(latt_info, HISQDirac(latt_info, 0.0, tol, maxiter, naik_epsilon, None))

        self.setForceParam(rational_param)
        self.quark = None  # For MultiHISQAction
        self.phi = LatticeStaggeredFermion(latt_info)
        self.eta = LatticeStaggeredFermion(latt_info)

        self.dirac.setPrecision(True)
        self.invert_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
        self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
        self.invert_param.solve_type = QudaSolveType.QUDA_DIRECT_PC_SOLVE
        self.invert_param.matpc_type = QudaMatPCType.QUDA_MATPC_EVEN_EVEN
        self.invert_param.mass_normalization = QudaMassNormalization.QUDA_MASS_NORMALIZATION

    def setForceParam(self, rational_param: RationalParam):
        coeff = [
            [
                2 * res,
                self.dirac.path_coeff_2[1] * 2 * res,
            ]
            for res in rational_param.residue_force
        ]
        if self.dirac.naik_epsilon != 0.0:
            coeff += [
                [
                    self.dirac.path_coeff_3[0] * self.dirac.naik_epsilon * 2 * res,
                    self.dirac.path_coeff_3[1] * self.dirac.naik_epsilon * 2 * res,
                ]
                for res in rational_param.residue_force
            ]
        self.num = len(rational_param.offset_force)
        self.num_naik = 0 if self.dirac.naik_epsilon == 0.0 else self.num
        self.coeff = numpy.array(coeff, "<f8")
        self.max_num_offset = max(
            len(rational_param.offset_force),
            len(rational_param.offset_sample),
            len(rational_param.offset_action),
        )
        self.rational_param = rational_param

    def updateFatLong(self):
        u_link = LatticeGauge(self.latt_info)
        saveGaugeQuda(u_link.data_ptrs, self.gauge_param)
        w_link = self.dirac.computeWLink(u_link)
        fatlink, longlink = self.dirac.computeXLink(w_link)
        fatlink, longlink = self.dirac.computeXLinkEpsilon(fatlink, longlink, w_link)
        self.dirac.loadFatLongGauge(fatlink, longlink)

    def updateFatLongReturn(self):
        u_link = LatticeGauge(self.latt_info)
        saveGaugeQuda(u_link.data_ptrs, self.gauge_param)
        v_link, w_link = self.dirac.computeVWLink(u_link)
        fatlink, longlink = self.dirac.computeXLink(w_link)
        fatlink, longlink = self.dirac.computeXLinkEpsilon(fatlink, longlink, w_link)
        self.dirac.loadFatLongGauge(fatlink, longlink)

        return u_link, v_link, w_link

    def sample(self):
        if self.quark is None:
            self.quark = MultiLatticeStaggeredFermion(self.latt_info, self.max_num_offset)
        self.sampleEta()
        self.updateFatLong()
        self.invertMultiShift("sample")

    def action(self) -> float:
        self.updateFatLong()
        self.invert_param.compute_action = 1
        self.invertMultiShift("force")
        self.invert_param.compute_action = 0
        return self.invert_param.action[0]

    def actionFA(self) -> float:
        self.updateFatLong()
        self.invertMultiShift("action")
        return self.eta.even.norm2()  # - norm_molecular_dynamics * self.phi.even.norm2()

    def force(self, dt, mom: Optional[LatticeMom] = None):
        assert self.quark is not None
        u_link, v_link, w_link = self.updateFatLongReturn()
        self.invertMultiShift("force")
        for i in range(self.num):
            dslashQuda(self.quark[i].odd_ptr, self.quark[i].even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
        if mom is not None:
            tmp = LatticeMom(mom.latt_info)
            self.gauge_param.use_resident_mom = 0
            self.gauge_param.make_resident_mom = 0
            self.gauge_param.return_result_mom = 1
        self.gauge_param.reconstruct = getGlobalReconstruct("staggered").cuda
        computeHISQForceQuda(
            nullptr if mom is None else tmp.data_ptrs,
            dt,
            self.dirac.path_coeff_2,
            self.dirac.path_coeff_1,
            w_link.data_ptrs,
            v_link.data_ptrs,
            u_link.data_ptrs,
            self.quark.data_ptrs,
            self.num,
            self.num_naik,
            self.coeff,
            self.gauge_param,
        )
        self.gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
        if mom is not None:
            self.gauge_param.use_resident_mom = 1
            self.gauge_param.make_resident_mom = 1
            self.gauge_param.return_result_mom = 0
            mom += tmp


class MultiHISQAction(StaggeredFermionAction):
    dirac: HISQDirac

    def __init__(self, latt_info: LatticeInfo, pseudo_fermions: List[HISQAction]) -> None:
        if latt_info.anisotropy != 1.0:
            getLogger().critical("anisotropy != 1.0 not implemented", NotImplementedError)
        super().__init__(latt_info, pseudo_fermions[0].dirac)
        self.pseudo_fermions = pseudo_fermions
        self.current_naik_epsilon = -1.0  # An impossible value for initialization

        self.setForceParam()
        self.quark = MultiLatticeStaggeredFermion(self.latt_info, self.max_num_offset)

    def setVerbosity(self, verbosity: QudaVerbosity):
        for pseudo_fermion in self.pseudo_fermions:
            pseudo_fermion.setVerbosity(verbosity)

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
        self.max_num_offset = max(
            [self.num] + [pseudo_fermion.max_num_offset for pseudo_fermion in self.pseudo_fermions]
        )

    def prepareFatLong(self):
        u_link = LatticeGauge(self.latt_info)
        saveGaugeQuda(u_link.data_ptrs, self.gauge_param)
        w_link = self.dirac.computeWLink(u_link)
        self.fatlink, self.longlink = self.dirac.computeXLink(w_link)
        self.w_link = w_link

    def prepareFatLongReturn(self):
        u_link = LatticeGauge(self.latt_info)
        saveGaugeQuda(u_link.data_ptrs, self.gauge_param)
        v_link, w_link = self.dirac.computeVWLink(u_link)
        self.fatlink, self.longlink = self.dirac.computeXLink(w_link)
        self.w_link = w_link

        return u_link, v_link, w_link

    def updateFatLong(self, pseudo_fermion: HISQAction):
        dirac = pseudo_fermion.dirac
        if self.current_naik_epsilon != dirac.naik_epsilon:
            fatlink, longlink = dirac.computeXLinkEpsilon(self.fatlink, self.longlink, self.w_link)
            dirac.loadFatLongGauge(fatlink, longlink)
            self.current_naik_epsilon = dirac.naik_epsilon

    def sample(self):
        self.prepareFatLong()
        for pseudo_fermion in self.pseudo_fermions:
            pseudo_fermion.sampleEta()
            self.updateFatLong(pseudo_fermion)
            pseudo_fermion.quark = self.quark
            pseudo_fermion.invertMultiShift("sample")

    def action(self) -> float:
        action = 0
        self.prepareFatLong()
        for pseudo_fermion in self.pseudo_fermions:
            self.updateFatLong(pseudo_fermion)
            pseudo_fermion.invert_param.compute_action = 1
            pseudo_fermion.quark = self.quark
            pseudo_fermion.invertMultiShift("force")
            pseudo_fermion.invert_param.compute_action = 0
            action += pseudo_fermion.invert_param.action[0]
        return action

    def actionFA(self) -> float:
        action = 0
        self.prepareFatLong()
        for pseudo_fermion in self.pseudo_fermions:
            self.updateFatLong(pseudo_fermion)
            pseudo_fermion.quark = self.quark
            pseudo_fermion.invertMultiShift("action")
            action += pseudo_fermion.eta.even.norm2()
        return action

    def force(self, dt, mom: Optional[LatticeMom] = None):
        assert self.quark is not None
        u_link, v_link, w_link = self.prepareFatLongReturn()
        num_current = 0
        for pseudo_fermion in self.pseudo_fermions:
            self.updateFatLong(pseudo_fermion)
            pseudo_fermion.quark = self.quark[num_current : num_current + pseudo_fermion.num]
            pseudo_fermion.invertMultiShift("force")
            for i in range(pseudo_fermion.num):
                dslashQuda(
                    self.quark[num_current + i].odd_ptr,
                    self.quark[num_current + i].even_ptr,
                    pseudo_fermion.invert_param,
                    QudaParity.QUDA_ODD_PARITY,
                )
            num_current += pseudo_fermion.num
        if mom is not None:
            tmp = LatticeMom(mom.latt_info)
            self.gauge_param.use_resident_mom = 0
            self.gauge_param.make_resident_mom = 0
            self.gauge_param.return_result_mom = 1
        self.gauge_param.reconstruct = getGlobalReconstruct("staggered").cuda
        computeHISQForceQuda(
            nullptr if mom is None else tmp.data_ptrs,
            dt,
            self.dirac.path_coeff_2,
            self.dirac.path_coeff_1,
            w_link.data_ptrs,
            v_link.data_ptrs,
            u_link.data_ptrs,
            self.quark.data_ptrs,
            self.num,
            self.num_naik,
            self.coeff,
            self.gauge_param,
        )
        self.gauge_param.reconstruct = QudaReconstructType.QUDA_RECONSTRUCT_NO
        if mom is not None:
            self.gauge_param.use_resident_mom = 1
            self.gauge_param.make_resident_mom = 1
            self.gauge_param.return_result_mom = 0
            mom += tmp
