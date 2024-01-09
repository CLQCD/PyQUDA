from abc import ABC, abstractmethod

from ..pointer import Pointer
from ..pyquda import QudaGaugeParam, QudaInvertParam, QudaMultigridParam
from ..field import LatticeInfo, LatticeGauge, LatticeFermion


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

    @abstractmethod
    def invert(self, b: LatticeFermion):
        pass
