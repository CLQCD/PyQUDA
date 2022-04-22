from typing import Sequence

import cupy as cp


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
        raise ValueError("Integer square root not defined for negative numbers.")


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
