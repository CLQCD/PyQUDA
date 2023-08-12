from abc import ABC, abstractmethod
from typing import List

from ..pointer import Pointer
from ..pyquda import QudaGaugeParam, QudaInvertParam, QudaMultigridParam
from ..field import LatticeGauge, LatticeFermion


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
    def newQudaMultigridParam(
        self,
        multigrid: bool,
        kappa: float,
        geo_block_size: List[List[int]],
        coarse_tol: float,
        coarse_maxiter: int,
        setup_tol: float,
        setup_maxiter: int,
        nu_pre: int,
        nu_post: int,
    ):
        pass

    @abstractmethod
    def newQudaInvertParam(self, kappa: float, tol: float, maxiter: float, *args, **kwargs):
        pass

    @abstractmethod
    def loadGauge(self, U: LatticeGauge):
        pass

    @abstractmethod
    def destroy(self):
        pass

    @abstractmethod
    def invert(self, b: LatticeFermion):
        pass
