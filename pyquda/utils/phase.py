from math import pi
from typing import List, NamedTuple, Sequence, Union

import numpy

from ..field import LatticeInfo


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


class MomentumPhaseOld:
    def __init__(self, latt_info: LatticeInfo) -> None:
        from .. import getCUDABackend

        backend = getCUDABackend()
        gx, gy, gz, gt = latt_info.grid_coord
        GLx, GLy, GLz, GLt = latt_info.global_size
        Lx, Ly, Lz, Lt = latt_info.size
        x = (2j * numpy.pi / GLx * numpy.arange(gx * Lx, (gx + 1) * Lx)).reshape(1, 1, Lx).repeat(Lz, 0).repeat(Ly, 1)
        y = (2j * numpy.pi / GLy * numpy.arange(gy * Ly, (gy + 1) * Ly)).reshape(1, Ly, 1).repeat(Lz, 0).repeat(Lx, 2)
        z = (2j * numpy.pi / GLz * numpy.arange(gz * Lz, (gz + 1) * Lz)).reshape(Lz, 1, 1).repeat(Ly, 1).repeat(Lx, 2)
        x_cb2 = numpy.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        y_cb2 = numpy.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        z_cb2 = numpy.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        for it in range(Lt):
            for iz in range(Lz):
                for iy in range(Ly):
                    ieo = (it + iz + iy) % 2
                    if ieo == 0:
                        x_cb2[0, it, iz, iy] = x[iz, iy, 0::2]
                        x_cb2[1, it, iz, iy] = x[iz, iy, 1::2]
                        y_cb2[0, it, iz, iy] = y[iz, iy, 0::2]
                        y_cb2[1, it, iz, iy] = y[iz, iy, 1::2]
                        z_cb2[0, it, iz, iy] = z[iz, iy, 0::2]
                        z_cb2[1, it, iz, iy] = z[iz, iy, 1::2]
                    else:
                        x_cb2[1, it, iz, iy] = x[iz, iy, 0::2]
                        x_cb2[0, it, iz, iy] = x[iz, iy, 1::2]
                        y_cb2[1, it, iz, iy] = y[iz, iy, 0::2]
                        y_cb2[0, it, iz, iy] = y[iz, iy, 1::2]
                        z_cb2[1, it, iz, iy] = z[iz, iy, 0::2]
                        z_cb2[0, it, iz, iy] = z[iz, iy, 1::2]

        if backend == "numpy":
            self.x = x_cb2
            self.y = y_cb2
            self.z = z_cb2
        elif backend == "cupy":
            import cupy

            self.x = cupy.asarray(x_cb2)
            self.y = cupy.asarray(y_cb2)
            self.z = cupy.asarray(z_cb2)
        elif backend == "torch":
            import torch

            self.x = torch.as_tensor(x_cb2)
            self.y = torch.as_tensor(y_cb2)
            self.z = torch.as_tensor(z_cb2)

    def getPhase(self, mom: Union[ThreeMomentumMode, Sequence[int]]):
        from .. import getCUDABackend

        backend = getCUDABackend()
        if not isinstance(mom, ThreeMomentumMode):
            mom = ThreeMomentumMode(*mom)
        if backend == "numpy":
            return numpy.exp(mom.x * self.x + mom.y * self.y + mom.z * self.z)
        elif backend == "cupy":
            import cupy

            return cupy.exp(mom.x * self.x + mom.y * self.y + mom.z * self.z)
        elif backend == "torch":
            import torch

            return torch.exp(mom.x * self.x + mom.y * self.y + mom.z * self.z)

    def getPhases(self, mom_list: Sequence[Union[ThreeMomentumMode, Sequence[int]]]):
        from .. import getCUDABackend

        backend = getCUDABackend()
        if backend == "numpy":
            phases = numpy.zeros((len(mom_list), *self.x.shape), "<c16")
        elif backend == "cupy":
            import cupy

            phases = cupy.zeros((len(mom_list), *self.x.shape), "<c16")
        elif backend == "torch":
            import torch

            phases = torch.zeros((len(mom_list), *self.x.shape), dtype=torch.complex128)
        for idx, mom in enumerate(mom_list):
            phases[idx] = self.getPhase(mom)

        return phases


class MomentumPhase:
    def __init__(self, latt_info: LatticeInfo) -> None:
        from .. import getCUDABackend

        backend = getCUDABackend()
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
        for iy in range(Ly):
            y[:, :, :, iy, :] = 2j * pi / GLy * (gy * Ly + iy)
        for iz in range(Lz):
            z[:, :, iz, :, :] = 2j * pi / GLz * (gz * Lz + iz)
        for it in range(Lt):
            t[:, it, :, :, :] = 2j * pi / GLt * (gt * Lt + it)

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

    def getPhase(self, mom: Union[ThreeMomentumMode, FourMomentumMode, Sequence[int]]):
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

    def getPhases(self, mom_list: Sequence[Union[ThreeMomentumMode, FourMomentumMode, Sequence[int]]]):
        from .. import getCUDABackend

        backend = getCUDABackend()
        if backend == "numpy":
            phases = numpy.zeros((len(mom_list), *self.x.shape), "<c16")
        elif backend == "cupy":
            import cupy

            phases = cupy.zeros((len(mom_list), *self.x.shape), "<c16")
        elif backend == "torch":
            import torch

            phases = torch.zeros((len(mom_list), *self.x.shape), dtype=torch.complex128)
        for idx, mom in enumerate(mom_list):
            phases[idx] = self.getPhase(mom)

        return phases


class Phase:
    def __init__(self, latt_info: LatticeInfo) -> None:
        from .. import getCUDABackend

        Gx, Gy, Gz, Gt = latt_info.grid_size
        gx, gy, gz, gt = latt_info.grid_coord
        Lx, Ly, Lz, Lt = latt_info.size
        x = numpy.arange(gx * Lx, (gx + 1) * Lx).reshape(1, 1, Lx).repeat(Lz, 0).repeat(Ly, 1) * (
            2j * numpy.pi / (Lx * Gx)
        )
        y = numpy.arange(gy * Ly, (gy + 1) * Ly).reshape(1, Ly, 1).repeat(Lz, 0).repeat(Lx, 2) * (
            2j * numpy.pi / (Ly * Gy)
        )
        z = numpy.arange(gz * Lz, (gz + 1) * Lz).reshape(Lz, 1, 1).repeat(Ly, 1).repeat(Lx, 2) * (
            2j * numpy.pi / (Lz * Gz)
        )
        x_cb2 = numpy.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        y_cb2 = numpy.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        z_cb2 = numpy.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        for it in range(Lt):
            for iz in range(Lz):
                for iy in range(Ly):
                    ieo = (it + iz + iy) % 2
                    if ieo == 0:
                        x_cb2[0, it, iz, iy] = x[iz, iy, 0::2]
                        x_cb2[1, it, iz, iy] = x[iz, iy, 1::2]
                        y_cb2[0, it, iz, iy] = y[iz, iy, 0::2]
                        y_cb2[1, it, iz, iy] = y[iz, iy, 1::2]
                        z_cb2[0, it, iz, iy] = z[iz, iy, 0::2]
                        z_cb2[1, it, iz, iy] = z[iz, iy, 1::2]
                    else:
                        x_cb2[1, it, iz, iy] = x[iz, iy, 0::2]
                        x_cb2[0, it, iz, iy] = x[iz, iy, 1::2]
                        y_cb2[1, it, iz, iy] = y[iz, iy, 0::2]
                        y_cb2[0, it, iz, iy] = y[iz, iy, 1::2]
                        z_cb2[1, it, iz, iy] = z[iz, iy, 0::2]
                        z_cb2[0, it, iz, iy] = z[iz, iy, 1::2]
        backend = getCUDABackend()
        if backend == "numpy":
            self.x = x_cb2
            self.y = y_cb2
            self.z = z_cb2
        elif backend == "cupy":
            import cupy

            self.x = cupy.asarray(x_cb2)
            self.y = cupy.asarray(y_cb2)
            self.z = cupy.asarray(z_cb2)
        elif backend == "torch":
            import torch

            self.x = torch.as_tensor(x_cb2)
            self.y = torch.as_tensor(y_cb2)
            self.z = torch.as_tensor(z_cb2)

    def __getitem__(self, momentum: List[int]):
        from .. import getCUDABackend

        npx, npy, npz = momentum
        backend = getCUDABackend()
        if backend == "numpy":
            return numpy.exp(npx * self.x + npy * self.y + npz * self.z)
        elif backend == "cupy":
            import cupy

            return cupy.exp(npx * self.x + npy * self.y + npz * self.z)
        elif backend == "torch":
            import torch

            return torch.exp(npx * self.x + npy * self.y + npz * self.z)

    def cache(self, mom_list: List[List[int]]):
        from .. import getCUDABackend

        backend = getCUDABackend()
        if backend == "numpy":
            ret = numpy.zeros((len(mom_list), *self.x.shape), "<c16")
        elif backend == "cupy":
            import cupy

            ret = cupy.zeros((len(mom_list), *self.x.shape), "<c16")
        elif backend == "torch":
            import torch

            ret = torch.zeros((len(mom_list), *self.x.shape), dtype=torch.complex128)
        for idx, mom in enumerate(mom_list):
            ret[idx] = self.__getitem__(mom)
        return ret
