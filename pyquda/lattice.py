from typing import List
from enum import IntEnum

import numpy as np

from .pyquda import getPointerArray, EvenPointer, OddPointer


class LatticeConstant(IntEnum):
    Nc = 3
    Nd = 4
    Ns = 4


Nc = LatticeConstant.Nc
Nd = LatticeConstant.Nd
Ns = LatticeConstant.Ns


def newLatticeFieldData(lattSize: List[int], dtype: str) -> np.ndarray:
    Lx, Ly, Lz, Lt = lattSize
    if dtype.capitalize() == "Gauge":
        return np.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
    elif dtype.capitalize() == "Fermion":
        return np.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c16")
    elif dtype.capitalize() == "Propagator":
        return np.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), "<c16")


class LatticeField:
    def __init__(self) -> None:
        pass


class LatticeGauge(LatticeField):
    def __init__(self, lattSize: List[int], value=None) -> None:
        self.lattSize = lattSize
        if value is None:
            self.data = newLatticeFieldData(lattSize, "Gauge").reshape(-1)
        else:
            self.data = value.reshape(-1)

    def setAntiPeroidicT(self):
        Lx, Ly, Lz, Lt = self.lattSize
        data = self.data.reshape(Nd, 2, Lt, -1)
        data[Nd - 1, :, Lt - 1] *= -1
        self.data = data.reshape(-1)

    @property
    def ptr(self):
        return getPointerArray(self.data.reshape(4, -1))


class LatticeFermion(LatticeField):
    def __init__(self, lattSize: List[int]) -> None:
        self.lattSize = lattSize
        self.data = newLatticeFieldData(lattSize, "Fermion").reshape(-1)

    @property
    def even(self):
        return self.data.reshape(2, -1)[0]

    @even.setter
    def even(self, value):
        data = self.data.reshape(2, -1)
        data[0] = value.reshape(-1)
        self.data = data.reshape(-1)

    @property
    def odd(self):
        return self.data.reshape(2, -1)[1]

    @odd.setter
    def odd(self, value):
        data = self.data.reshape(2, -1)
        data[1] = value.reshape(-1)
        self.data = data.reshape(-1)

    @property
    def even_ptr(self):
        return EvenPointer(self.data.reshape(2, -1))

    @property
    def odd_ptr(self):
        return OddPointer(self.data.reshape(2, -1))


class LatticePropagator(LatticeField):
    def __init__(self, lattSize: List[int]) -> None:
        self.lattSize = lattSize
        self.data = newLatticeFieldData(lattSize, "Propagator").reshape(-1)
