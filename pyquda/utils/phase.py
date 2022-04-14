from typing import Sequence

import cupy as cp


class Phase:
    def __init__(self, latt_size: Sequence[int]) -> None:
        Lx, Ly, Lz, Lt = latt_size
        x = cp.arange(Lx).reshape(1, 1, 1, Lx).repeat(Lt, 0).repeat(Lz, 1).repeat(Ly, 2) * (2j * cp.pi / Lx)
        y = cp.arange(Ly).reshape(1, 1, Ly, 1).repeat(Lt, 0).repeat(Lz, 1).repeat(Lx, 3) * (2j * cp.pi / Ly)
        z = cp.arange(Lz).reshape(1, Lz, 1, 1).repeat(Lt, 0).repeat(Ly, 2).repeat(Lx, 3) * (2j * cp.pi / Lz)
        self.x = cp.ones((2, Lt, Lz, Ly, Lx // 2), "<c16")
        self.y = cp.ones((2, Lt, Lz, Ly, Lx // 2), "<c16")
        self.z = cp.ones((2, Lt, Lz, Ly, Lx // 2), "<c16")
        for it in range(Lt):
            for iz in range(Lz):
                for iy in range(Ly):
                    ieo = (it + iz + iy) % 2
                    if ieo == 0:
                        self.x[0, it, iz, iy] = x[it, iz, iy, 0::2]
                        self.x[1, it, iz, iy] = x[it, iz, iy, 1::2]
                        self.y[0, it, iz, iy] = y[it, iz, iy, 0::2]
                        self.y[1, it, iz, iy] = y[it, iz, iy, 1::2]
                        self.z[0, it, iz, iy] = z[it, iz, iy, 0::2]
                        self.z[1, it, iz, iy] = z[it, iz, iy, 1::2]
                    else:
                        self.x[1, it, iz, iy] = x[it, iz, iy, 0::2]
                        self.x[0, it, iz, iy] = x[it, iz, iy, 1::2]
                        self.y[1, it, iz, iy] = y[it, iz, iy, 0::2]
                        self.y[0, it, iz, iy] = y[it, iz, iy, 1::2]
                        self.z[1, it, iz, iy] = z[it, iz, iy, 0::2]
                        self.z[0, it, iz, iy] = z[it, iz, iy, 1::2]

    def __getitem__(self, momentum: Sequence[int]):
        npx, npy, npz = momentum
        return cp.exp(npx * self.x + npy * self.y + npz * self.z)

    def cache(self, mom_list: Sequence[Sequence[int]]):
        ret = []
        for mom in mom_list:
            ret.append(self.__getitem__(mom))
        return ret
