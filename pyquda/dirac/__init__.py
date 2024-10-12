from abc import ABC, abstractmethod
from typing import List, NamedTuple, Union

from ..pointer import Pointer
from ..pyquda import (
    QudaGaugeParam,
    QudaInvertParam,
    QudaMultigridParam,
    QudaGaugeSmearParam,
    QudaGaugeObservableParam,
    invertQuda,
    invertMultiSrcQuda,
    invertMultiShiftQuda,
    MatQuda,
    MatDagMatQuda,
    dslashQuda,
    dslashMultiSrcQuda,
    newMultigridQuda,
    updateMultigridQuda,
    destroyMultigridQuda,
)
from ..enum_quda import (
    QUDA_MAX_MULTI_SHIFT,
    QudaBoolean,
    QudaDagType,
    QudaMatPCType,
    QudaParity,
    QudaPrecision,
    QudaReconstructType,
    QudaSolverNormalization,
    QudaVerbosity,
)
from ..field import (
    LatticeInfo,
    LatticeGauge,
    LatticeFermion,
    MultiLatticeFermion,
    LatticeStaggeredFermion,
    MultiLatticeStaggeredFermion,
)


class Precision(NamedTuple):
    cpu: QudaPrecision
    cuda: QudaPrecision
    sloppy: QudaPrecision
    precondition: QudaPrecision
    eigensolver: QudaPrecision


class Reconstruct(NamedTuple):
    cuda: QudaReconstructType
    sloppy: QudaReconstructType
    precondition: QudaReconstructType
    eigensolver: QudaReconstructType


_precision = Precision(
    QudaPrecision.QUDA_DOUBLE_PRECISION,
    QudaPrecision.QUDA_DOUBLE_PRECISION,
    QudaPrecision.QUDA_HALF_PRECISION,
    QudaPrecision.QUDA_HALF_PRECISION,
    QudaPrecision.QUDA_SINGLE_PRECISION,
)
_reconstruct = Reconstruct(
    QudaReconstructType.QUDA_RECONSTRUCT_12,
    QudaReconstructType.QUDA_RECONSTRUCT_12,
    QudaReconstructType.QUDA_RECONSTRUCT_12,
    QudaReconstructType.QUDA_RECONSTRUCT_12,
)


def setPrecision(
    *,
    cuda: QudaPrecision = None,
    sloppy: QudaPrecision = None,
    precondition: QudaPrecision = None,
    eigensolver: QudaPrecision = None,
):
    global _precision
    _precision = Precision(
        _precision.cpu,
        cuda if cuda is not None else _precision.cuda,
        sloppy if sloppy is not None else _precision.sloppy,
        precondition if precondition is not None else _precision.precondition,
        eigensolver if eigensolver is not None else _precision.eigensolver,
    )


def setReconstruct(
    *,
    cuda: QudaReconstructType = None,
    sloppy: QudaReconstructType = None,
    precondition: QudaReconstructType = None,
    eigensolver: QudaReconstructType = None,
):
    global _reconstruct
    _reconstruct = Reconstruct(
        cuda if cuda is not None else _reconstruct.cuda,
        sloppy if sloppy is not None else _reconstruct.sloppy,
        precondition if precondition is not None else _reconstruct.precondition,
        eigensolver if eigensolver is not None else _reconstruct.eigensolver,
    )


class Gauge(ABC):
    latt_info: LatticeInfo
    precision: Precision
    reconstruct: Reconstruct
    gauge_param: QudaGaugeParam
    invert_param: QudaInvertParam
    smear_param: QudaGaugeSmearParam
    obs_param: QudaGaugeObservableParam

    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info
        self.precision = Precision(
            _precision.cpu, _precision.cuda, _precision.sloppy, _precision.precondition, _precision.eigensolver
        )
        self.reconstruct = Reconstruct(
            _reconstruct.cuda, _reconstruct.sloppy, _reconstruct.precondition, _reconstruct.eigensolver
        )

    def _setPrecision(
        self,
        *,
        cuda: QudaPrecision = None,
        sloppy: QudaPrecision = None,
        precondition: QudaPrecision = None,
        eigensolver: QudaPrecision = None,
    ):
        self.precision = Precision(
            self.precision.cpu,
            cuda if cuda is not None else self.precision.cuda,
            sloppy if sloppy is not None else self.precision.sloppy,
            precondition if precondition is not None else self.precision.precondition,
            eigensolver if eigensolver is not None else self.precision.eigensolver,
        )

    def _setReconstruct(
        self,
        *,
        cuda: QudaReconstructType = None,
        sloppy: QudaReconstructType = None,
        precondition: QudaReconstructType = None,
        eigensolver: QudaReconstructType = None,
    ):
        self.reconstruct = Reconstruct(
            cuda if cuda is not None else self.reconstruct.cuda,
            sloppy if sloppy is not None else self.reconstruct.sloppy,
            precondition if precondition is not None else self.reconstruct.precondition,
            eigensolver if eigensolver is not None else self.reconstruct.eigensolver,
        )


class Multigrid:
    param: QudaMultigridParam
    inv_param: QudaInvertParam
    instance: Pointer

    def __init__(self, param: QudaMultigridParam, inv_param: QudaInvertParam) -> None:
        self.param = param
        self.inv_param = inv_param
        self.instance = None

    def new(self):
        if self.instance is not None:
            destroyMultigridQuda(self.instance)
        self.instance = newMultigridQuda(self.param)

    def update(self, thin_update_only: bool):
        if self.instance is not None:
            if thin_update_only:
                self.param.thin_update_only = QudaBoolean.QUDA_BOOLEAN_TRUE
                updateMultigridQuda(self.instance, self.param)
                self.param.thin_update_only = QudaBoolean.QUDA_BOOLEAN_FALSE
            else:
                updateMultigridQuda(self.instance, self.param)

    def destroy(self):
        if self.instance is not None:
            destroyMultigridQuda(self.instance)
        self.instance = None


class Dirac(Gauge):
    multigrid: Multigrid

    def __init__(self, latt_info: LatticeInfo) -> None:
        super().__init__(latt_info)

    @abstractmethod
    def loadGauge(self, gauge: LatticeGauge):
        pass

    @abstractmethod
    def destroy(self):
        pass

    def performance(self):
        from .. import getLogger

        gflops, secs = self.invert_param.gflops, self.invert_param.secs
        if self.invert_param.verbosity >= QudaVerbosity.QUDA_SUMMARIZE:
            getLogger().info(f"Time = {secs:.3f} secs, Performance = {gflops / secs:.3f} GFLOPS")

    def invert(self, b: LatticeFermion):
        x = LatticeFermion(b.latt_info)
        invertQuda(x.data_ptr, b.data_ptr, self.invert_param)
        self.performance()
        return x

    def invertRestart(self, b: LatticeFermion, restart: int):
        x = self.invert(b)
        self.invert_param.solver_normalization = QudaSolverNormalization.QUDA_SOURCE_NORMALIZATION
        for _ in range(restart):
            r = b - self.mat(x)
            x += self.invert(r)
        self.invert_param.solver_normalization = QudaSolverNormalization.QUDA_DEFAULT_NORMALIZATION
        return x

    def mat(self, x: LatticeFermion):
        b = LatticeFermion(x.latt_info)
        MatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def matDagMat(self, x: LatticeFermion):
        b = LatticeFermion(x.latt_info)
        MatDagMatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def dslash(self, x: LatticeFermion, parity: QudaParity):
        b = LatticeFermion(x.latt_info)
        dslashQuda(b.data_ptr, x.data_ptr, self.invert_param, parity)
        return b

    def invertMultiSrc(self, b: MultiLatticeFermion):
        self.invert_param.num_src = b.L5
        x = MultiLatticeFermion(b.latt_info, b.L5)
        invertMultiSrcQuda(x.data_ptrs, b.data_ptrs, self.invert_param)
        self.performance()
        return x

    def dslashMultiSrc(self, x: MultiLatticeFermion, parity: QudaParity):
        self.invert_param.num_src = x.L5
        b = MultiLatticeFermion(x.latt_info, x.L5)
        dslashMultiSrcQuda(b.data_ptrs, x.data_ptrs, self.invert_param, parity)
        return b

    def _invertMultiShiftParam(self, offset: List[float], residue: List[float], norm: float = None):
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
        return num_offset

    def _invertMultiShiftPC(
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
                self.performance()
                if norm is not None:
                    x.even = norm * b.even
                    for i in range(num_offset):
                        x.even += residue[i] * xx[i].even
            else:
                if norm is None:
                    invertQuda(xx[0].even_ptr, b.even_ptr, self.invert_param)
                    self.performance()
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
                self.performance()
                if norm is not None:
                    x.odd = norm * b.odd
                    for i in range(num_offset):
                        x.odd += residue[i] * xx[i].odd
            else:
                if norm is None:
                    invertQuda(xx[0].odd_ptr, b.odd_ptr, self.invert_param)
                    self.performance()
                else:
                    self.invert_param.dagger = QudaDagType.QUDA_DAG_YES
                    MatQuda(x.odd_ptr, b.odd_ptr, self.invert_param)
                    self.invert_param.dagger = QudaDagType.QUDA_DAG_NO

    def invertMultiShiftPC(
        self, b: LatticeFermion, offset: List[float], residue: List[float], norm: float = None
    ) -> Union[LatticeFermion, MultiLatticeFermion]:
        num_offset = self._invertMultiShiftParam(offset, residue, norm)
        xx = MultiLatticeFermion(self.latt_info, num_offset) if norm is None or num_offset > 1 else None
        x = LatticeFermion(self.latt_info) if norm is not None else None
        self._invertMultiShiftPC(xx, x, b, residue, norm)
        return xx if norm is None else x

    def newMultigrid(self):
        if self.multigrid.param is not None:
            self.multigrid.new()
            self.invert_param.preconditioner = self.multigrid.instance

    def updateMultigrid(self, thin_update_only: bool):
        if self.multigrid.param is not None:
            self.multigrid.update(thin_update_only)
            self.invert_param.preconditioner = self.multigrid.instance

    def destroyMultigrid(self):
        if self.multigrid.param is not None:
            self.multigrid.destroy()


class StaggeredDirac(Dirac):
    def invert(self, b: LatticeStaggeredFermion):
        x = LatticeStaggeredFermion(b.latt_info)
        invertQuda(x.data_ptr, b.data_ptr, self.invert_param)
        self.performance()
        return x

    def invertRestart(self, b: LatticeStaggeredFermion, restart: int):
        x = self.invert(b)
        for _ in range(restart):
            r = b - self.mat(x)
            x += self.invert(r)
        return x

    def mat(self, x: LatticeStaggeredFermion):
        b = LatticeStaggeredFermion(x.latt_info)
        MatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def matDagMat(self, x: LatticeStaggeredFermion):
        b = LatticeStaggeredFermion(x.latt_info)
        MatDagMatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def dslash(self, x: LatticeStaggeredFermion, parity: QudaParity):
        b = LatticeStaggeredFermion(x.latt_info)
        dslashQuda(b.data_ptr, x.data_ptr, self.invert_param, parity)
        return b

    def invertMultiSrc(self, b: MultiLatticeStaggeredFermion):
        self.invert_param.num_src = b.L5
        x = MultiLatticeStaggeredFermion(b.latt_info, b.L5)
        invertMultiSrcQuda(x.data_ptrs, b.data_ptrs, self.invert_param)
        self.performance()
        return x

    def dslashMultiSrc(self, x: MultiLatticeStaggeredFermion, parity: QudaParity):
        self.invert_param.num_src = x.L5
        b = MultiLatticeStaggeredFermion(x.latt_info, x.L5)
        dslashMultiSrcQuda(b.data_ptrs, x.data_ptrs, self.invert_param, parity)
        return b

    def invertMultiShiftPC(
        self, b: LatticeStaggeredFermion, offset: List[float], residue: List[float], norm: float = None
    ) -> Union[LatticeStaggeredFermion, MultiLatticeStaggeredFermion]:
        num_offset = self._invertMultiShiftParam(offset, residue, norm)
        xx = MultiLatticeStaggeredFermion(self.latt_info, num_offset) if norm is None or num_offset > 1 else None
        x = LatticeStaggeredFermion(self.latt_info) if norm is not None else None
        self._invertMultiShiftPC(xx, x, b, residue, norm)
        return xx if norm is None else x
