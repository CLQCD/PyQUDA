from abc import ABC, abstractmethod

from ..pointer import Pointer
from ..pyquda import QudaGaugeParam, QudaInvertParam, QudaMultigridParam, QudaGaugeSmearParam, QudaGaugeObservableParam
from ..field import LatticeInfo, LatticeGauge, LatticeFermion, LatticeStaggeredFermion

from . import general
from .general import Precision, Reconstruct


class Gauge(ABC):
    latt_info: LatticeInfo
    precision: Precision
    reconstruct: Reconstruct
    gauge_param: QudaGaugeParam
    smear_param: QudaGaugeSmearParam
    obs_param: QudaGaugeObservableParam

    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = LatticeInfo(latt_info.global_size)
        self.precision = Precision()
        self.reconstruct = Reconstruct()


class Dirac(ABC):
    latt_info: LatticeInfo
    precision: Precision
    reconstruct: Reconstruct
    gauge_param: QudaGaugeParam
    invert_param: QudaInvertParam
    mg_param: QudaMultigridParam
    mg_inv_param: QudaInvertParam
    mg_instance: Pointer

    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info
        self.precision = Precision()
        self.reconstruct = Reconstruct()

    @abstractmethod
    def loadGauge(self, gauge: LatticeGauge):
        pass

    @abstractmethod
    def destroy(self):
        pass

    def invert(self, b: LatticeFermion):
        return general.invert(b, self.invert_param)

    def mat(self, x: LatticeFermion):
        return general.mat(x, self.invert_param)

    def matDagMat(self, x: LatticeFermion):
        return general.matDagMat(x, self.invert_param)


class StaggeredDirac(Dirac):
    def invert(self, b: LatticeStaggeredFermion):
        return general.invertStaggered(b, self.invert_param)

    def mat(self, x: LatticeStaggeredFermion):
        return general.matStaggered(x, self.invert_param)

    def matDagMat(self, x: LatticeStaggeredFermion):
        return general.matDagMatStaggered(x, self.invert_param)
