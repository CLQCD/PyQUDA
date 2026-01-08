from abc import ABC, abstractmethod
from typing import List, Literal, NamedTuple, Optional

from pyquda_comm.array import arrayRandomNormalComplex
from ..field import (
    LatticeInfo,
    LatticeMom,
    LatticeFermion,
    LatticeStaggeredFermion,
    MultiLatticeFermion,
    MultiLatticeStaggeredFermion,
)
from ..enum_quda import QUDA_MAX_MULTI_SHIFT, QudaDagType, QudaSolutionType, QudaVerbosity
from ..quda import QudaGaugeParam, QudaInvertParam, invertQuda, MatQuda, invertMultiShiftQuda
from ..dirac import getGlobalPrecision
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
    norm_force: float = 0.0
    residue_force: List[float] = [1.0]
    offset_force: List[float] = [0.0]
    norm_sample: float = 0.0
    residue_sample: List[float] = [1.0]
    offset_sample: List[float] = [0.0]
    norm_action: float = 0.0
    residue_action: List[float] = [1.0]
    offset_action: List[float] = [0.0]


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

    def setVerbosity(self, verbosity: QudaVerbosity):
        self.dirac.setVerbosity(verbosity)

    @abstractmethod
    def sample(self):
        pass

    def sampleEta(self):
        self.eta.data = arrayRandomNormalComplex(0.0, 2.0**-0.5, self.eta.shape, self.eta.location)

    def invertMultiShift(self, mode: Literal["sample", "action", "force"]):
        if mode == "sample":
            offset, residue, norm = (
                self.rational_param.offset_sample,
                self.rational_param.residue_sample,
                self.rational_param.norm_sample,
            )
            xx, x, b = self.quark, self.phi, self.eta
        elif mode == "action":
            offset, residue, norm = (
                self.rational_param.offset_action,
                self.rational_param.residue_action,
                self.rational_param.norm_action,
            )
            xx, x, b = self.quark, self.eta, self.phi
        elif mode == "force":
            offset, residue, norm = (
                self.rational_param.offset_force,
                self.rational_param.residue_force,
                self.rational_param.norm_force,
            )
            xx, x, b = self.quark, self.eta, self.phi
        assert len(offset) == len(residue)
        num_offset = len(offset)
        if num_offset > 1:
            tol = self.invert_param.tol
            use_invert_sloppy = tol > 2**-15 and (
                min(map(abs, residue)) / max(map(abs, residue)) > 2**-8
            )  # ? tol_0 > epsilon_short and tol_min / tol_max > epsilon_float / epsilon_short
            self.invert_param.num_offset = num_offset
            self.invert_param.offset = offset + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
            self.invert_param.residue = residue + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
            self.invert_param.tol_offset = [
                max(tol * abs(residue[0] / residue[i]), 2e-16 * self.latt_info.GLt**0.5) for i in range(num_offset)
            ] + [0.0] * (QUDA_MAX_MULTI_SHIFT - num_offset)
            if use_invert_sloppy:
                self.invert_param.cuda_prec_sloppy = getGlobalPrecision("invert").sloppy
            invertMultiShiftQuda(xx.even_ptrs, b.even_ptr, self.invert_param)
            self.dirac.performance()
            if use_invert_sloppy:
                self.invert_param.cuda_prec_sloppy = getGlobalPrecision("multishift").sloppy
            if mode == "sample" or mode == "action":
                x.even = norm * b.even
                for i in range(num_offset):
                    x.even += residue[i] * xx[i].even
        else:  # S = -\phi^\dagger (M^\dagger M)^{-1} \phi
            if mode == "sample":
                self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
                MatQuda(x.even_ptr, b.even_ptr, self.invert_param)
                self.invert_param.dagger = QudaDagType.QUDA_DAG_NO
            elif mode == "action":
                self.invert_param.solution_type = QudaSolutionType.QUDA_MATPC_SOLUTION
                self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
                invertQuda(x.even_ptr, b.even_ptr, self.invert_param)
                self.dirac.performance()
                self.invert_param.solution_type = QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
                self.invert_param.dagger = QudaDagType.QUDA_DAG_NO
            elif mode == "force":
                invertQuda(xx[0].even_ptr, b.even_ptr, self.invert_param)
                self.dirac.performance()


class StaggeredFermionAction(FermionAction):
    dirac: StaggeredFermionDirac
    quark: Optional[MultiLatticeStaggeredFermion]
    phi: LatticeStaggeredFermion
    eta: LatticeStaggeredFermion

    def __init__(self, latt_info: LatticeInfo, dirac: StaggeredFermionDirac) -> None:
        super().__init__(latt_info, dirac)
