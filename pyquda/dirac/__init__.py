from abc import ABC, abstractmethod
from typing import NamedTuple

from ..pointer import Pointer
from ..pyquda import (
    QudaGaugeParam,
    QudaInvertParam,
    QudaMultigridParam,
    QudaGaugeSmearParam,
    QudaGaugeObservableParam,
    invertQuda,
    MatQuda,
    MatDagMatQuda,
    newMultigridQuda,
    updateMultigridQuda,
    destroyMultigridQuda,
)
from ..enum_quda import QudaBoolean, QudaPrecision, QudaReconstructType
from ..field import LatticeInfo, LatticeGauge, LatticeFermion, LatticeStaggeredFermion


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
        getLogger().info(f"Time = {secs:.3f} secs, Performance = {gflops / secs:.3f} GFLOPS")

    def invert(self, b: LatticeFermion):
        x = LatticeFermion(b.latt_info)
        invertQuda(x.data_ptr, b.data_ptr, self.invert_param)
        self.performance()
        return x

    def mat(self, x: LatticeFermion):
        b = LatticeFermion(x.latt_info)
        MatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def matDagMat(self, x: LatticeFermion):
        b = LatticeFermion(x.latt_info)
        MatDagMatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

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

    def mat(self, x: LatticeStaggeredFermion):
        b = LatticeStaggeredFermion(x.latt_info)
        MatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b

    def matDagMat(self, x: LatticeStaggeredFermion):
        b = LatticeStaggeredFermion(x.latt_info)
        MatDagMatQuda(b.data_ptr, x.data_ptr, self.invert_param)
        return b
