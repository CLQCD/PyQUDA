from abc import ABC, abstractmethod

from ..pointer import Pointer
from ..pyquda import QudaGaugeParam, QudaInvertParam, QudaMultigridParam
from ..field import LatticeInfo, LatticeGauge, LatticeFermion, LatticeStaggeredFermion

from . import general


class Dirac(ABC):
    latt_info: LatticeInfo
    gauge_param: QudaGaugeParam
    invert_param: QudaInvertParam
    mg_param: QudaMultigridParam
    mg_inv_param: QudaInvertParam
    mg_instance: Pointer

    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info

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
