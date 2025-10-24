from abc import ABC, abstractmethod
from typing import List, Literal, NamedTuple, Optional, Union, overload

from pyquda_comm import getArrayBackend
from pyquda_comm.array import arrayRandomNormalComplex
from ..field import (
    LatticeInfo,
    LatticeMom,
    LatticeFermion,
    LatticeStaggeredFermion,
    MultiLatticeFermion,
    MultiLatticeStaggeredFermion,
)
from ..enum_quda import QUDA_MAX_MULTI_SHIFT, QudaDagType, QudaMatPCType
from ..pyquda import QudaGaugeParam, QudaInvertParam, invertQuda, MatQuda, invertMultiShiftQuda
from ..dirac.abstract import Dirac, FermionDirac, StaggeredFermionDirac


class LoopParam(NamedTuple):
    path: List[List[int]]
    coeff: List[float]


class Action(ABC):
    latt_info: LatticeInfo
    dirac: Dirac
    gauge_param: QudaGaugeParam
    loop_param: LoopParam

    def __init__(self, latt_info: LatticeInfo, dirac: Dirac) -> None:
        self.latt_info = latt_info
        self.dirac = dirac
        self.gauge_param = self.dirac.gauge_param

    @abstractmethod
    def action(self) -> float:
        pass

    @abstractmethod
    def force(self, dt: float, mom: Optional[LatticeMom] = None):
        pass


class RationalParam(NamedTuple):
    norm_molecular_dynamics: float = 0.0
    residue_molecular_dynamics: List[float] = [1.0]
    offset_molecular_dynamics: List[float] = [0.0]
    norm_pseudo_fermion: float = 0.0
    residue_pseudo_fermion: List[float] = [1.0]
    offset_pseudo_fermion: List[float] = [0.0]
    norm_fermion_action: float = 0.0
    residue_fermion_action: List[float] = [1.0]
    offset_fermion_action: List[float] = [0.0]


class FermionAction(Action):
    dirac: FermionDirac
    invert_param: QudaInvertParam
    rational_param: RationalParam
    quark: MultiLatticeFermion
    phi: LatticeFermion
    eta: LatticeFermion

    def __init__(self, latt_info: LatticeInfo, dirac: FermionDirac) -> None:
        super().__init__(latt_info, dirac)
        self.invert_param = self.dirac.invert_param

    @abstractmethod
    def sample(self):
        pass

    def sampleEta(self):
        self.eta.data[:] = arrayRandomNormalComplex(0.0, 2.0**-0.5, self.eta.shape, getArrayBackend())

    def _invertMultiShiftParam(self, mode: Literal["pseudo_fermion", "molecular_dynamics", "fermion_action"]):
        if mode == "pseudo_fermion":
            offset, residue, norm = (
                self.rational_param.offset_pseudo_fermion,
                self.rational_param.residue_pseudo_fermion,
                self.rational_param.norm_pseudo_fermion,
            )
        elif mode == "molecular_dynamics":
            offset, residue, norm = (
                self.rational_param.offset_molecular_dynamics,
                self.rational_param.residue_molecular_dynamics,
                None,
            )
        elif mode == "fermion_action":
            offset, residue, norm = (
                self.rational_param.offset_fermion_action,
                self.rational_param.residue_fermion_action,
                self.rational_param.norm_fermion_action,
            )
        assert len(offset) == len(residue)
        num_offset = len(offset)
        if num_offset > 1:
            tol = self.invert_param.tol
            self.invert_param.num_offset = num_offset
            self.invert_param.offset = offset + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
            self.invert_param.residue = residue + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
            self.invert_param.tol_offset = [
                max(tol * abs(residue[0] / residue[i]), 2e-16 * (self.latt_info.Gt * self.latt_info.Lt) ** 0.5)
                for i in range(num_offset)
            ] + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
        else:
            assert offset == [0.0] and residue == [1.0] and (norm is None or norm == 0.0)
        return residue, norm

    @overload
    def _invertMultiShift(
        self,
        xx: MultiLatticeFermion,
        x: LatticeFermion,
        b: LatticeFermion,
        residue: List[float],
        norm: Optional[float],
    ): ...

    @overload
    def _invertMultiShift(
        self,
        xx: MultiLatticeStaggeredFermion,
        x: LatticeStaggeredFermion,
        b: LatticeStaggeredFermion,
        residue: List[float],
        norm: Optional[float],
    ): ...

    def _invertMultiShift(
        self,
        xx: Union[MultiLatticeFermion, MultiLatticeStaggeredFermion],
        x: Union[LatticeFermion, LatticeStaggeredFermion],
        b: Union[LatticeFermion, LatticeStaggeredFermion],
        residue: List[float],
        norm: Optional[float],
    ):
        num_offset = len(residue)
        if (
            self.invert_param.matpc_type == QudaMatPCType.QUDA_MATPC_EVEN_EVEN
            or self.invert_param.matpc_type == QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
        ):
            if num_offset > 1:
                invertMultiShiftQuda(xx.even_ptrs, b.even_ptr, self.invert_param)
                self.dirac.performance()
                if norm is not None:
                    x.data[0] = norm * b.data[0]  # x.even = norm * b.even
                    for i in range(num_offset):
                        x.data[0] += residue[i] * xx[i].data[0]  # x.even += residue[i] * xx[i].even
            else:
                if norm is None:
                    invertQuda(xx[0].even_ptr, b.even_ptr, self.invert_param)
                    self.dirac.performance()
                else:
                    self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
                    MatQuda(x.even_ptr, b.even_ptr, self.invert_param)
                    self.invert_param.dagger = QudaDagType.QUDA_DAG_NO
        elif (
            self.invert_param.matpc_type == QudaMatPCType.QUDA_MATPC_ODD_ODD
            or self.invert_param.matpc_type == QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC
        ):
            if num_offset > 1:
                invertMultiShiftQuda(xx.odd_ptrs, b.odd_ptr, self.invert_param)
                self.dirac.performance()
                if norm is not None:
                    x.data[1] = norm * b.data[1]  # x.odd = norm * b.odd
                    for i in range(num_offset):
                        x.data[1] += residue[i] * xx[i].data[1]  # x.odd += residue[i] * xx[i].odd
            else:
                if norm is None:
                    invertQuda(xx[0].odd_ptr, b.odd_ptr, self.invert_param)
                    self.dirac.performance()
                else:
                    self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
                    MatQuda(x.odd_ptr, b.odd_ptr, self.invert_param)
                    self.invert_param.dagger = QudaDagType.QUDA_DAG_NO

    def invertMultiShift(self, mode: Literal["pseudo_fermion", "molecular_dynamics", "fermion_action"]):
        residue, norm = self._invertMultiShiftParam(mode)
        if mode == "pseudo_fermion":
            self._invertMultiShift(self.quark, self.phi, self.eta, residue, norm)
        else:
            self._invertMultiShift(self.quark, self.eta, self.phi, residue, norm)


class StaggeredFermionAction(FermionAction):
    dirac: StaggeredFermionDirac
    quark: Optional[MultiLatticeStaggeredFermion]
    phi: LatticeStaggeredFermion
    eta: LatticeStaggeredFermion

    def __init__(self, latt_info: LatticeInfo, dirac: StaggeredFermionDirac) -> None:
        super().__init__(latt_info, dirac)
