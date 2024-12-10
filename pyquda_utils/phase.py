from math import pi, isqrt
from typing import Sequence

import numpy

from .core import getLogger, getCUDABackend, evenodd, LatticeInfo


def getMomList(mom2_max, mom2_min=0):
    mom_list = []
    radius = isqrt(mom2_max)
    for npz in range(-radius, radius + 1):
        for npy in range(-radius, radius + 1):
            for npx in range(-radius, radius + 1):
                np2 = npx**2 + npy**2 + npz**2
                if np2 <= mom2_max and np2 >= mom2_min:
                    mom_list.append((npx, npy, npz))
    return mom_list


def getMomDict(mom2_max, mom2_min=0):
    mom_list = getMomList(mom2_max, mom2_min)
    mom_dict = {key: " ".join([str(np) for np in val]) for key, val in enumerate(mom_list)}
    return mom_dict


class MomentumPhase:
    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info
        gx, gy, gz, gt = latt_info.grid_coord
        Lx, Ly, Lz, Lt = latt_info.size

        x = numpy.zeros((4, 2, Lt, Lz, Ly, Lx // 2), "<c16")
        xx = numpy.arange(gx * Lx, (gx + 1) * Lx)
        for it in range(Lt):
            for iz in range(Lz):
                for iy in range(Ly):
                    ieo = (it + iz + iy) % 2
                    if ieo == 0:
                        x[0, 0, it, iz, iy] = xx[0::2]
                        x[0, 1, it, iz, iy] = xx[1::2]
                    else:
                        x[0, 1, it, iz, iy] = xx[0::2]
                        x[0, 0, it, iz, iy] = xx[1::2]
        x[1] = numpy.arange(gy * Ly, (gy + 1) * Ly).reshape(1, 1, 1, Ly, 1)
        x[2] = numpy.arange(gz * Lz, (gz + 1) * Lz).reshape(1, 1, Lz, 1, 1)
        x[3] = numpy.arange(gt * Lt, (gt + 1) * Lt).reshape(1, Lt, 1, 1, 1)

        backend = getCUDABackend()
        if backend == "numpy":
            self.x = x
        elif backend == "cupy":
            import cupy

            self.x = cupy.asarray(x)
        elif backend == "torch":
            import torch

            self.x = torch.as_tensor(x)

    def getPhase(self, mom_mode: Sequence[int], x0: Sequence[int] = [0, 0, 0, 0]):
        x = self.x
        global_size = self.latt_info.global_size

        if len(mom_mode) == 3:
            ip = [2j * pi * mom_mode[i] / global_size[i] for i in range(3)]
            ipx = ip[0] * x[0] + ip[1] * x[1] + ip[2] * x[2]
            ipx0 = ip[0] * x0[0] + ip[1] * x0[1] + ip[2] * x0[2]
        elif len(mom_mode) == 4:
            ip = [2j * pi * mom_mode[i] / global_size[i] for i in range(4)]
            ipx = ip[0] * x[0] + ip[1] * x[1] + ip[2] * x[2] + ip[3] * x[3]
            ipx0 = ip[0] * x0[0] + ip[1] * x0[1] + ip[2] * x0[2] + ip[3] * x0[3]
        else:
            getLogger().critical(f"mom should be a sequence of int with length 3 or 4, but get {mom_mode}", ValueError)

        backend = getCUDABackend()
        if backend == "numpy":
            return numpy.exp(ipx - ipx0)
        elif backend == "cupy":
            import cupy

            return cupy.exp(ipx - ipx0)
        elif backend == "torch":
            import torch

            return torch.exp(ipx - ipx0)

    def getPhases(self, mom_mode_list: Sequence[Sequence[int]], x0: Sequence[int] = [0, 0, 0, 0]):
        Lx, Ly, Lz, Lt = self.latt_info.size

        backend = getCUDABackend()
        if backend == "numpy":
            phases = numpy.zeros((len(mom_mode_list), 2, Lt, Lz, Ly, Lx // 2), "<c16")
        elif backend == "cupy":
            import cupy

            phases = cupy.zeros((len(mom_mode_list), 2, Lt, Lz, Ly, Lx // 2), "<c16")
        elif backend == "torch":
            import torch

            phases = torch.zeros((len(mom_mode_list), 2, Lt, Lz, Ly, Lx // 2), dtype=torch.complex128)
        for idx, mom in enumerate(mom_mode_list):
            phases[idx] = self.getPhase(mom, x0)

        return phases


class GridPhase:
    def __init__(self, latt_info: LatticeInfo, stride: Sequence[int]) -> None:
        self.latt_info = latt_info
        self.stride = stride

    def getPhase(self, t_srce: Sequence[int]):
        gx, gy, gz, gt = self.latt_info.grid_coord
        Lx, Ly, Lz, Lt = self.latt_info.size
        Sx, Sy, Sz, St = self.stride
        x, y, z, t = t_srce
        sx, sy, sz, st = (x + gx * Lx) % Sx, (y + gy * Ly) % Sy, (z + gz * Lz) % Sz, (t + gt * Lt) % St
        phase = numpy.zeros((Lt, Lz, Ly, Lx), "<i4")
        if sx < Lx and sy < Ly and sz < Lz and st < Lt:
            phase[st::St, sz::Sz, sy::Sy, sx::Sx] = 1
        phase = evenodd(phase, [0, 1, 2, 3])

        backend = getCUDABackend()
        if backend == "numpy":
            return phase
        elif backend == "cupy":
            import cupy

            return cupy.asarray(phase)
        elif backend == "torch":
            import torch

            return torch.as_tensor(phase)

    def getPhases(self, t_srce_list: Sequence[Sequence[int]]):
        Lx, Ly, Lz, Lt = self.latt_info.size

        backend = getCUDABackend()
        if backend == "numpy":
            phases = numpy.zeros((len(t_srce_list), 2, Lt, Lz, Ly, Lx // 2), "<c16")
        elif backend == "cupy":
            import cupy

            phases = cupy.zeros((len(t_srce_list), 2, Lt, Lz, Ly, Lx // 2), "<c16")
        elif backend == "torch":
            import torch

            phases = torch.zeros((len(t_srce_list), 2, Lt, Lz, Ly, Lx // 2), dtype=torch.complex128)
        for idx, t_srce in enumerate(t_srce_list):
            phases[idx] = self.getPhase(t_srce)

        return phases
