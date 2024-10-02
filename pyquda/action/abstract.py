from abc import ABC, abstractmethod

from ..field import LatticeInfo, LatticeFermion, LatticeStaggeredFermion


class GaugeAction(ABC):
    latt_info: LatticeInfo

    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info

    @abstractmethod
    def action(self) -> float:
        pass

    @abstractmethod
    def force(self, dt: float):
        pass


class FermionAction(ABC):
    latt_info: LatticeInfo

    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info

    @abstractmethod
    def action(self, new_gauge: bool) -> float:
        pass

    @abstractmethod
    def force(self, dt: float, new_gauge: bool):
        pass

    @abstractmethod
    def sample(self, noise: LatticeFermion, new_gauge: bool):
        pass


class StaggeredFermionAction(ABC):
    latt_info: LatticeInfo

    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info

    @abstractmethod
    def action(self, new_gauge: bool) -> float:
        pass

    @abstractmethod
    def force(self, dt: float, new_gauge: bool):
        pass

    @abstractmethod
    def sample(self, noise: LatticeStaggeredFermion, new_gauge: bool):
        pass
