from math import pi
from typing import NamedTuple, Sequence

import numpy

from ..field import LatticeInfo, cb2


def isqrt(n):
    if n > 0:
        x = 1 << (n.bit_length() + 1 >> 1)
        while True:
            y = (x + n // x) >> 1
            if y >= x:
                return x
            x = y
    elif n == 0:
        return 0
    else:
        from .. import getLogger

        getLogger().critical("Integer square root not defined for negative numbers", ValueError)


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


class ThreeMomentumMode(NamedTuple):
    x: int
    y: int
    z: int


class FourMomentumMode(NamedTuple):
    x: int
    y: int
    z: int
    t: int


class MomentumPhase:
    def __init__(self, latt_info: LatticeInfo) -> None:
        from .. import getCUDABackend

        self.latt_info = latt_info
        gx, gy, gz, gt = latt_info.grid_coord
        GLx, GLy, GLz, GLt = latt_info.global_size
        Lx, Ly, Lz, Lt = latt_info.size

        x = numpy.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        y = numpy.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        z = numpy.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        t = numpy.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")

        xx = 2j * pi / GLx * numpy.arange(gx * Lx, (gx + 1) * Lx)
        for it in range(Lt):
            for iz in range(Lz):
                for iy in range(Ly):
                    ieo = (it + iz + iy) % 2
                    if ieo == 0:
                        x[0, it, iz, iy] = xx[0::2]
                        x[1, it, iz, iy] = xx[1::2]
                    else:
                        x[1, it, iz, iy] = xx[0::2]
                        x[0, it, iz, iy] = xx[1::2]
        y[:] = 2j * pi / GLy * numpy.arange(gy * Ly, (gy + 1) * Ly).reshape(1, 1, 1, Ly, 1)
        z[:] = 2j * pi / GLz * numpy.arange(gz * Lz, (gz + 1) * Lz).reshape(1, 1, Lz, 1, 1)
        t[:] = 2j * pi / GLz * numpy.arange(gt * Lt, (gt + 1) * Lt).reshape(1, Lt, 1, 1, 1)

        backend = getCUDABackend()
        if backend == "numpy":
            self.x = x
            self.y = y
            self.z = z
            self.t = t
        elif backend == "cupy":
            import cupy

            self.x = cupy.asarray(x)
            self.y = cupy.asarray(y)
            self.z = cupy.asarray(z)
            self.t = cupy.asarray(t)
        elif backend == "torch":
            import torch

            self.x = torch.as_tensor(x)
            self.y = torch.as_tensor(y)
            self.z = torch.as_tensor(z)
            self.t = torch.as_tensor(t)

    def getPhase(self, mom: Sequence[int]):
        from .. import getCUDABackend

        backend = getCUDABackend()
        if not isinstance(mom, ThreeMomentumMode) or not isinstance(mom, FourMomentumMode):
            if len(mom) == 3:
                mom = ThreeMomentumMode(*mom)
            elif len(mom) == 4:
                mom = FourMomentumMode(*mom)
            else:
                from .. import getLogger

                getLogger().critical(f"mom should be a sequence of int with length 3 or 4, but get {mom}", ValueError)
        if isinstance(mom, ThreeMomentumMode):
            if backend == "numpy":
                return numpy.exp(mom.x * self.x + mom.y * self.y + mom.z * self.z)
            elif backend == "cupy":
                import cupy

                return cupy.exp(mom.x * self.x + mom.y * self.y + mom.z * self.z)
            elif backend == "torch":
                import torch

                return torch.exp(mom.x * self.x + mom.y * self.y + mom.z * self.z)
        elif isinstance(mom, FourMomentumMode):
            if backend == "numpy":
                return numpy.exp(mom.x * self.x + mom.y * self.y + mom.z * self.z + mom.t * self.t)
            elif backend == "cupy":
                import cupy

                return cupy.exp(mom.x * self.x + mom.y * self.y + mom.z * self.z + mom.t * self.t)
            elif backend == "torch":
                import torch

                return torch.exp(mom.x * self.x + mom.y * self.y + mom.z * self.z + mom.t * self.t)

    def getPhases(self, mom_list: Sequence[Sequence[int]]):
        from .. import getCUDABackend

        Lx, Ly, Lz, Lt = self.latt_info.size

        backend = getCUDABackend()
        if backend == "numpy":
            phases = numpy.zeros((len(mom_list), 2, Lt, Lz, Ly, Lx // 2), "<c16")
        elif backend == "cupy":
            import cupy

            phases = cupy.zeros((len(mom_list), 2, Lt, Lz, Ly, Lx // 2), "<c16")
        elif backend == "torch":
            import torch

            phases = torch.zeros((len(mom_list), 2, Lt, Lz, Ly, Lx // 2), dtype=torch.complex128)
        for idx, mom in enumerate(mom_list):
            phases[idx] = self.getPhase(mom)

        return phases


class GridPhase:
    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info

    def getPhase(self, stride: Sequence[int]):
        from .. import getCUDABackend

        gx, gy, gz, gt = self.latt_info.grid_coord
        Lx, Ly, Lz, Lt = self.latt_info.size
        Sx, Sy, Sz, St = stride
        phase = numpy.zeros((Lt, Lz, Ly, Lx), "<i4")
        if (gt * Lt) % St < Lt and (gz * Lz) % Sz < Lz and (gy * Ly) % Sy < Ly and (gx * Lx) % Sx < Lx:
            phase[(gt * Lt) % St :: St, (gz * Lz) % Sz :: Sz, (gy * Ly) % Sy :: Sy, (gx * Lx) % Sx :: Sx] = 1
        phase = cb2(phase, [0, 1, 2, 3])

        backend = getCUDABackend()
        if backend == "numpy":
            return phase
        elif backend == "cupy":
            import cupy

            return cupy.asarray(phase)
        elif backend == "torch":
            import torch

            return torch.as_tensor(phase)

    def getPhases(self, stride_list: Sequence[Sequence[int]]):
        from .. import getCUDABackend

        Lx, Ly, Lz, Lt = self.latt_info.size

        backend = getCUDABackend()
        if backend == "numpy":
            phases = numpy.zeros((len(stride_list), 2, Lt, Lz, Ly, Lx // 2), "<c16")
        elif backend == "cupy":
            import cupy

            phases = cupy.zeros((len(stride_list), 2, Lt, Lz, Ly, Lx // 2), "<c16")
        elif backend == "torch":
            import torch

            phases = torch.zeros((len(stride_list), 2, Lt, Lz, Ly, Lx // 2), dtype=torch.complex128)
        for idx, stride in enumerate(stride_list):
            phases[idx] = self.getPhase(stride)

        return phases
