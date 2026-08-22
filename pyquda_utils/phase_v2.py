from math import pi
from typing import Sequence

import numpy

from pyquda_comm.array import arrayAsArray, arrayDType, arrayExp, arrayOnes, arrayZeros
from pyquda_comm.field import LatticeInt, MultiLatticeInt, LatticeComplex, MultiLatticeComplex
from .core import getArrayBackend, LatticeInfo


class LocationPhase:
    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info
        self.x = arrayAsArray(latt_info.coordinate(), getArrayBackend())

    def getPhase(self):
        return MultiLatticeInt(self.latt_info, self.latt_info.Nd, self.x)


class DistancePhase:
    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info
        self.x = arrayAsArray(latt_info.coordinate(), getArrayBackend())

    def getPhase(self, x0: Sequence[int]):
        phase = MultiLatticeInt(self.latt_info, self.latt_info.Nd)
        for i in range(self.latt_info.Nd):
            GL = self.latt_info.global_size[i]
            phase[i].data = (self.x[i] - x0[i] + GL // 2) % GL - GL // 2
        return phase


class MomentumPhase:
    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info
        self.x = arrayAsArray(latt_info.coordinate(), getArrayBackend())

    def getPhase(self, mom: Sequence[int], x0: Sequence[int] = [0, 0, 0, 0]):
        backend = getArrayBackend()
        ipx = arrayZeros(self.x[0].shape, arrayDType("<c16", backend), backend)
        for i in range(len(mom)):
            ipx += 2j * pi * mom[i] / self.latt_info.global_size[i] * (self.x[i] - x0[i])
        return LatticeComplex(self.latt_info, arrayExp(ipx, backend))

    def getPhases(self, mom_mode_list: Sequence[Sequence[int]], x0: Sequence[int] = [0, 0, 0, 0]):
        phases = MultiLatticeComplex(self.latt_info, len(mom_mode_list))
        for idx, mom_mode in enumerate(mom_mode_list):
            phases[idx] = self.getPhase(mom_mode, x0)
        return phases


class GridPhase:
    def __init__(self, latt_info: LatticeInfo, stride: Sequence[int]) -> None:
        self.latt_info = latt_info
        self.stride = stride
        self.x = arrayAsArray(latt_info.coordinate(), getArrayBackend())

    def getPhase(self, t_srce: Sequence[int]):
        backend = getArrayBackend()
        # sx, sy, sz, st = (x + gx * Lx) % Sx, (y + gy * Ly) % Sy, (z + gz * Lz) % Sz, (t + gt * Lt) % St
        phase = arrayOnes(self.x[0].shape, arrayDType("<i4", backend), backend)
        for i in range(self.latt_info.Nd):
            phase &= (self.x[i] >= t_srce[i]) & ((self.x[i] - t_srce[i]) % self.stride[i] == 0)
        return LatticeInt(self.latt_info, phase)

    def getPhases(self, t_srce_list: Sequence[Sequence[int]]):
        phases = MultiLatticeInt(self.latt_info, len(t_srce_list))
        for idx, t_srce in enumerate(t_srce_list):
            phases[idx] = self.getPhase(t_srce)
        return phases
