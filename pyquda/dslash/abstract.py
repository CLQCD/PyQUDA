from abc import ABC, abstractmethod
from typing import List

from ..pyquda import Pointer, QudaGaugeParam, QudaInvertParam, QudaMultigridParam
from ..core import LatticeGauge, LatticeFermion


class Dslash(ABC):
    gauge_param: QudaGaugeParam
    invert_param: QudaInvertParam
    mg_param: QudaMultigridParam
    mg_inv_param: QudaInvertParam
    mg_instance: Pointer

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def newQudaGaugeParam(self, latt_size: List[int], anisotropy: float, t_boundary: int):
        pass

    @abstractmethod
    def newQudaInvertParam(self, kappa: float, tol: float, maxiter: float, *args, **kwargs):
        pass

    @abstractmethod
    def loadGauge(self, U: LatticeGauge):
        pass

    @abstractmethod
    def invert(self, b: LatticeFermion):
        pass

    # @abstractmethod
    # def newQudaMultigridParam(self):
    #     pass
