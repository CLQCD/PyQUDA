from abc import ABC, abstractmethod
from functools import partial
from typing import List, Literal, NamedTuple, Union

from .. import getCUDABackend
from ..enum_quda import QUDA_MAX_MULTI_SHIFT, QudaDagType, QudaMatPCType
from ..pyquda import QudaGaugeParam, QudaInvertParam, invertQuda, MatQuda, invertMultiShiftQuda
from ..dirac.abstract import Gauge, Dirac, StaggeredDirac
from ..field import (
    LatticeInfo,
    LatticeFermion,
    LatticeStaggeredFermion,
    MultiLatticeFermion,
    MultiLatticeStaggeredFermion,
)


class RHMCParam(NamedTuple):
    norm_molecular_dynamics: float = 0.0
    residue_molecular_dynamics: List[float] = [1.0]
    offset_molecular_dynamics: List[float] = [0.0]
    norm_pseudo_fermion: float = 0.0
    residue_pseudo_fermion: List[float] = [1.0]
    offset_pseudo_fermion: List[float] = [0.0]
    norm_fermion_action: float = 0.0
    residue_fermion_action: List[float] = [1.0]
    offset_fermion_action: List[float] = [0.0]


class GaugeAction(ABC):
    latt_info: LatticeInfo
    dirac: Gauge
    gauge_param: QudaGaugeParam

    def __init__(self, latt_info: LatticeInfo, dirac: Gauge) -> None:
        self.latt_info = latt_info
        self.dirac = dirac
        self.gauge_param = self.dirac.gauge_param
        self.is_fermion = False
        self.is_staggered = False

    @abstractmethod
    def action(self) -> float:
        pass

    @abstractmethod
    def force(self, dt: float):
        pass


class FermionAction(GaugeAction):
    dirac: Dirac
    invert_param: QudaInvertParam
    rhmc_param: RHMCParam
    phi: LatticeFermion
    eta: LatticeFermion

    def __init__(self, latt_info: LatticeInfo, dirac: Dirac) -> None:
        super().__init__(latt_info, dirac)
        self.invert_param = self.dirac.invert_param
        self.is_fermion = True

    def sampleEta(self):
        def _backend():
            backend = getCUDABackend()
            if backend == "numpy":
                import numpy

                return numpy, numpy.random.random
            elif backend == "cupy":
                import cupy

                return cupy, partial(cupy.random.random, dtype=cupy.float64)
            elif backend == "torch":
                import torch

                return torch, partial(torch.rand, dtype=torch.float64)

        def _normal(backend, random, shape):
            theta = 2 * backend.pi * random(shape)
            r = backend.sqrt(-backend.log(random(shape)))
            z = r * (backend.cos(theta) + 1j * backend.sin(theta))
            return z

        backend, random = _backend()
        self.eta.data = _normal(backend, random, self.phi.shape)

    @abstractmethod
    def sample(self, new_gauge: bool):
        pass

    @abstractmethod
    def action(self, new_gauge: bool) -> float:
        pass

    @abstractmethod
    def force(self, dt: float, new_gauge: bool):
        pass

    def _invertMultiShiftParam(self, mode: Literal["pseudo_fermion", "molecular_dynamics", "fermion_action"]):
        if mode == "pseudo_fermion":
            offset, residue, norm = (
                self.rhmc_param.offset_pseudo_fermion,
                self.rhmc_param.residue_pseudo_fermion,
                self.rhmc_param.norm_pseudo_fermion,
            )
        elif mode == "molecular_dynamics":
            offset, residue, norm = (
                self.rhmc_param.offset_molecular_dynamics,
                self.rhmc_param.residue_molecular_dynamics,
                None,
            )
        elif mode == "fermion_action":
            offset, residue, norm = (
                self.rhmc_param.offset_fermion_action,
                self.rhmc_param.residue_fermion_action,
                self.rhmc_param.norm_fermion_action,
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
        return num_offset, residue, norm

    def _invertMultiShift(
        self,
        xx: Union[MultiLatticeFermion, MultiLatticeStaggeredFermion],
        x: Union[LatticeFermion, LatticeStaggeredFermion],
        b: Union[LatticeFermion, LatticeStaggeredFermion],
        residue: List[float],
        norm: float,
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
                    x.even = norm * b.even
                    for i in range(num_offset):
                        x.even += residue[i] * xx[i].even
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
                    x.odd = norm * b.odd
                    for i in range(num_offset):
                        x.odd += residue[i] * xx[i].odd
            else:
                if norm is None:
                    invertQuda(xx[0].odd_ptr, b.odd_ptr, self.invert_param)
                    self.dirac.performance()
                else:
                    self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
                    MatQuda(x.odd_ptr, b.odd_ptr, self.invert_param)
                    self.invert_param.dagger = QudaDagType.QUDA_DAG_NO

    def invertMultiShift(
        self, mode: Literal["pseudo_fermion", "molecular_dynamics", "fermion_action"]
    ) -> Union[LatticeFermion, MultiLatticeFermion]:
        num_offset, residue, norm = self._invertMultiShiftParam(mode)
        xx = MultiLatticeFermion(self.latt_info, num_offset) if norm is None or num_offset > 1 else None
        if mode == "pseudo_fermion":
            self._invertMultiShift(xx, self.phi, self.eta, residue, norm)
        else:
            self._invertMultiShift(xx, self.eta, self.phi, residue, norm)
        return xx


class StaggeredFermionAction(FermionAction):
    dirac: StaggeredDirac
    phi: LatticeStaggeredFermion
    eta: LatticeStaggeredFermion

    def __init__(self, latt_info: LatticeInfo, dirac: StaggeredDirac) -> None:
        super().__init__(latt_info, dirac)
        self.is_staggered = True

    @abstractmethod
    def sample(self, noise: LatticeStaggeredFermion, new_gauge: bool):
        pass

    @abstractmethod
    def action(self, new_gauge: bool) -> float:
        pass

    @abstractmethod
    def force(self, dt: float, new_gauge: bool):
        pass

    def invertMultiShift(
        self, mode: Literal["pseudo_fermion", "molecular_dynamics", "fermion_action"]
    ) -> Union[LatticeStaggeredFermion, MultiLatticeStaggeredFermion]:
        num_offset, residue, norm = self._invertMultiShiftParam(mode)
        xx = MultiLatticeStaggeredFermion(self.latt_info, num_offset) if norm is None or num_offset > 1 else None
        if mode == "pseudo_fermion":
            self._invertMultiShift(xx, self.phi, self.eta, residue, norm)
        else:
            self._invertMultiShift(xx, self.eta, self.phi, residue, norm)
        return xx
