from typing import List
from enum import IntEnum

import numpy
import cupy
import torch

from .pyquda import ndarrayDataPointer


class LatticeConstant(IntEnum):
    Nc = 3
    Nd = 4
    Ns = 4


Nc = LatticeConstant.Nc
Nd = LatticeConstant.Nd
Ns = LatticeConstant.Ns


def lexico(data: numpy.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Np, Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    assert Np == 2, "There must be 2 parities."
    Lx *= 2
    Npre = int(numpy.prod(shape[:axes[0]]))
    Nsuf = int(numpy.prod(shape[axes[-1] + 1:]))
    dtype = data.dtype if dtype is None else dtype
    data_cb2 = data.reshape(Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf)
    data_lexico = numpy.zeros((Npre, Lt, Lz, Ly, Lx, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_lexico[:, t, z, y, 0::2] = data_cb2[:, 0, t, z, y, :]
                    data_lexico[:, t, z, y, 1::2] = data_cb2[:, 1, t, z, y, :]
                else:
                    data_lexico[:, t, z, y, 1::2] = data_cb2[:, 0, t, z, y, :]
                    data_lexico[:, t, z, y, 0::2] = data_cb2[:, 1, t, z, y, :]
    return data_lexico.reshape(*shape[:axes[0]], Lt, Lz, Ly, Lx, *shape[axes[-1] + 1:])


def cb2(data: numpy.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    Npre = int(numpy.prod(shape[:axes[0]]))
    Nsuf = int(numpy.prod(shape[axes[-1] + 1:]))
    dtype = data.dtype if dtype is None else dtype
    data_lexico = data.reshape(Npre, Lt, Lz, Ly, Lx, Nsuf)
    data_cb2 = numpy.zeros((Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_cb2[:, 0, t, z, y, :] = data_lexico[:, t, z, y, 0::2]
                    data_cb2[:, 1, t, z, y, :] = data_lexico[:, t, z, y, 1::2]
                else:
                    data_cb2[:, 0, t, z, y, :] = data_lexico[:, t, z, y, 1::2]
                    data_cb2[:, 1, t, z, y, :] = data_lexico[:, t, z, y, 0::2]
    return data_cb2.reshape(*shape[:axes[0]], 2, Lt, Lz, Ly, Lx // 2, *shape[axes[-1] + 1:])


def newLatticeFieldData(latt_size: List[int], dtype: str) -> cupy.ndarray:
    Lx, Ly, Lz, Lt = latt_size
    if dtype.capitalize() == "Gauge":
        ret = cupy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
        ret[:] = cupy.identity(Nc)
        return ret
    elif dtype.capitalize() == "Colorvector":
        return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
    elif dtype.capitalize() == "Fermion":
        return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), dtype="<c16")
        # return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), dtype=torch.complex128, device="cuda")
    elif dtype.capitalize() == "Propagator":
        return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), "<c16")
        # return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), dtype=torch.complex128, device="cuda")


class LatticeField:
    def __init__(self) -> None:
        pass


class LatticeGauge(LatticeField):
    def __init__(self, latt_size: List[int], value=None, t_boundary=True) -> None:
        Lx, Ly, Lz, Lt = latt_size
        self.latt_size = latt_size
        if value is None:
            self.data = newLatticeFieldData(latt_size, "Gauge")
        else:
            self.data = value.reshape(Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc)
        self.t_boundary = t_boundary

    def copy(self):
        res = LatticeGauge(self.latt_size, None, self.t_boundary)
        res.data[:] = self.data[:]
        return res

    def setAntiPeroidicT(self):
        if self.t_boundary:
            Lt = self.latt_size[Nd - 1]
            data = self.data.reshape(Nd, 2, Lt, -1)
            data[Nd - 1, :, Lt - 1] *= -1

    def setAnisotropy(self, anisotropy: float):
        data = self.data.reshape(Nd, -1)
        data[:Nd - 1] /= anisotropy

    @property
    def data_ptr(self):
        return ndarrayDataPointer(self.data.reshape(4, -1), True)

    @property
    def data_ptrs(self):
        return ndarrayDataPointer(self.data.reshape(4, -1), True)

    def lexico(self):
        return lexico(self.data.get(), [1, 2, 3, 4, 5])


class LatticeColorVector(LatticeField):
    def __init__(self, latt_size: List[int], value=None) -> None:
        Lx, Ly, Lz, Lt = latt_size
        self.latt_size = latt_size
        if value is None:
            self.data = newLatticeFieldData(latt_size, "Colorvector")
        else:
            self.data = value.reshape(2, Lt, Lz, Ly, Lx // 2, Nc)

    @property
    def even(self):
        return self.data[0]

    @even.setter
    def even(self, value):
        self.data[0] = value

    @property
    def odd(self):
        return self.data[1]

    @odd.setter
    def odd(self, value):
        self.data[1] = value

    @property
    def data_ptr(self):
        return ndarrayDataPointer(self.data.reshape(-1), True)

    @property
    def even_ptr(self):
        return ndarrayDataPointer(self.data.reshape(2, -1)[0], True)

    @property
    def odd_ptr(self):
        return ndarrayDataPointer(self.data.reshape(2, -1)[1], True)

    def lexico(self):
        return lexico(self.data.get(), [0, 1, 2, 3, 4])


class LatticeFermion(LatticeField):
    def __init__(self, latt_size: List[int], value=None) -> None:
        Lx, Ly, Lz, Lt = latt_size
        self.latt_size = latt_size
        if value is None:
            self.data = newLatticeFieldData(latt_size, "Fermion")
        else:
            self.data = value.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)

    @property
    def even(self):
        return self.data[0]

    @even.setter
    def even(self, value):
        self.data[0] = value

    @property
    def odd(self):
        return self.data[1]

    @odd.setter
    def odd(self, value):
        self.data[1] = value

    @property
    def data_ptr(self):
        return ndarrayDataPointer(self.data.reshape(-1), True)

    @property
    def even_ptr(self):
        return ndarrayDataPointer(self.data.reshape(2, -1)[0], True)

    @property
    def odd_ptr(self):
        return ndarrayDataPointer(self.data.reshape(2, -1)[1], True)

    def lexico(self):
        return lexico(self.data.get(), [0, 1, 2, 3, 4])


class LatticePropagator(LatticeField):
    def __init__(self, latt_size: List[int]) -> None:
        self.latt_size = latt_size
        self.data = newLatticeFieldData(latt_size, "Propagator")

    def lexico(self):
        return lexico(self.data.get(), [0, 1, 2, 3, 4])

    def transpose(self):
        return self.data.transpose(0, 1, 2, 3, 4, 6, 5, 8, 7).copy()
