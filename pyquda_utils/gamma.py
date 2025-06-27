from typing import Union

import numpy

from pyquda import getCUDABackend
from pyquda_comm.array import arrayDevice
from pyquda_comm.field import LatticePropagator


class GammaMatrix:
    gamma_0 = numpy.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        "<c16",
    )
    gamma_1 = numpy.array(
        [
            [0, 0, 0, 1j],
            [0, 0, 1j, 0],
            [0, -1j, 0, 0],
            [-1j, 0, 0, 0],
        ],
        "<c16",
    )
    gamma_2 = numpy.array(
        [
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
        ],
        "<c16",
    )
    gamma_3 = numpy.array(
        [
            [0, 0, 1j, 0],
            [0, 0, 0, -1j],
            [-1j, 0, 0, 0],
            [0, 1j, 0, 0],
        ],
        "<c16",
    )
    gamma_4 = numpy.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        "<c16",
    )

    @classmethod
    def matrix(cls, index):
        gamma = (
            (cls.gamma_1 if index & 0b0001 else cls.gamma_0)
            @ (cls.gamma_2 if index & 0b0010 else cls.gamma_0)
            @ (cls.gamma_3 if index & 0b0100 else cls.gamma_0)
            @ (cls.gamma_4 if index & 0b1000 else cls.gamma_0)
        )
        backend = getCUDABackend()
        return arrayDevice(gamma, backend)


class GammaSparse:
    gamma_indices = [
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [0, 1, 2, 3],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [0, 1, 2, 3],
    ]
    gamma_data = [
        [1, 1, 1, 1],
        [1j, 1j, -1j, -1j],
        [-1, 1, 1, -1],
        [-1j, 1j, -1j, 1j],
        [1j, -1j, -1j, 1j],
        [-1, 1, -1, 1],
        [-1j, -1j, -1j, -1j],
        [1, 1, -1, -1],
        [1, 1, 1, 1],
        [1j, 1j, -1j, -1j],
        [-1, 1, 1, -1],
        [-1j, 1j, -1j, 1j],
        [1j, -1j, -1j, 1j],
        [-1, 1, -1, 1],
        [-1j, -1j, -1j, -1j],
        [1, 1, -1, -1],
    ]

    @classmethod
    def indices(cls, index):
        return cls.gamma_indices[index]

    @classmethod
    def data(cls, index):
        return cls.gamma_data[index]


class Gamma:
    popcnt = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]
    popsign = [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1]

    def __init__(self, index: int, factor: float = 1) -> None:
        assert isinstance(index, int) and 0b0000 <= index <= 0b1111, "index should be int from 0 to 15"
        self.index = index
        self.factor = factor

    def __repr__(self) -> str:
        return (
            f"{'-' if self.factor == -1 else '' if self.factor == 1 else self.factor}"
            f"{'γ₀' if not self.index else ''}"
            f"{'γ₁' if self.index & 0b0001 else ''}"
            f"{'γ₂' if self.index & 0b0010 else ''}"
            f"{'γ₃' if self.index & 0b0100 else ''}"
            f"{'γ₄' if self.index & 0b1000 else ''}"
        )

    def __neg__(self) -> "Gamma":
        return Gamma(self.index, -self.factor)

    def __mul__(self, rhs: Union[int, float, complex]) -> "Gamma":
        if not isinstance(rhs, (int, float, complex)):
            return NotImplemented
        return Gamma(self.index, self.factor * rhs)

    def __rmul__(self, lhs: Union[int, float, complex]) -> "Gamma":
        if not isinstance(lhs, (int, float, complex)):
            return NotImplemented
        return Gamma(self.index, lhs * self.factor)

    def __truediv__(self, rhs: Union[int, float, complex]) -> "Gamma":
        if not isinstance(rhs, (int, float, complex)):
            return NotImplemented
        return Gamma(self.index, self.factor / rhs)

    def __add__(self, rhs: "Gamma") -> Union["Gamma", "Polarize"]:
        if not isinstance(rhs, Gamma):
            return NotImplemented
        if self.index == rhs.index:
            return Gamma(self.index, self.factor + rhs.factor)
        else:
            return Polarize(self, rhs)

    def __sub__(self, rhs: "Gamma") -> Union["Gamma", "Polarize"]:
        if not isinstance(rhs, Gamma):
            return NotImplemented
        if self.index == rhs.index:
            return Gamma(self.index, self.factor - rhs.factor)
        else:
            return Polarize(self, -rhs)

    def __matmul__(self, rhs: "Gamma") -> "Gamma":
        if not isinstance(rhs, Gamma):
            return NotImplemented
        index = self.index ^ rhs.index
        factor = self.factor * rhs.factor
        if self.index & 0b1000:
            factor *= Gamma.popsign[rhs.index & 0b0111]
        if self.index & 0b0100:
            factor *= Gamma.popsign[rhs.index & 0b0011]
        if self.index & 0b0010:
            factor *= Gamma.popsign[rhs.index & 0b0001]
        # if self.index & 0b0001:
        #     factor *= Gamma.popsign[rhs.index & 0b0000]
        return Gamma(index, factor)

    @property
    def T(self) -> "Gamma":
        factor = self.factor
        factor *= -1 if Gamma.popcnt[self.index] & 0b10 else 1
        factor *= Gamma.popsign[self.index & 0b0101]
        return Gamma(self.index, factor)

    @property
    def H(self) -> "Gamma":
        factor = self.factor.conjugate()
        factor *= -1 if Gamma.popcnt[self.index] & 0b10 else 1
        return Gamma(self.index, factor)

    @property
    def matrix(self):
        return self.factor * GammaMatrix.matrix(self.index)

    @property
    def sparse_indices(self):
        return GammaSparse.indices(self.index)

    @property
    def sparse_data(self):
        return GammaSparse.data(self.index)


class Polarize:
    def __init__(self, left: Gamma, right: Gamma) -> None:
        assert isinstance(left, Gamma), "left should be Gamma"
        assert isinstance(right, Gamma), "right should be Gamma"
        assert left.index != right.index, "left and right should be different"
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return (
            f"{self.left} + {self.right}"
            if self.right.factor > 0
            else f"{self.left} - {Gamma(self.right.index, -self.right.factor)}"
        )

    def __neg__(self) -> "Polarize":
        return Polarize(-self.left, -self.right)

    def __mul__(self, rhs: Union[int, float, complex]) -> "Gamma":
        if not isinstance(rhs, (int, float, complex)):
            return NotImplemented
        return Polarize(self.left * rhs, self.right * rhs)

    def __rmul__(self, lhs: Union[int, float, complex]) -> "Gamma":
        if not isinstance(lhs, (int, float, complex)):
            return NotImplemented
        return Polarize(lhs * self.left, lhs * self.right)

    def __truediv__(self, rhs: Union[int, float, complex]) -> "Gamma":
        if not isinstance(rhs, (int, float, complex)):
            return NotImplemented
        return Polarize(self.left / rhs, self.right / rhs)

    def __matmul__(self, rhs: Gamma) -> "Polarize":
        if not isinstance(rhs, Gamma):
            return NotImplemented
        return Polarize(self.left @ rhs, self.right @ rhs)

    def __rmatmul__(self, lhs: Gamma) -> "Polarize":
        if not isinstance(lhs, Gamma):
            return NotImplemented
        return Polarize(lhs @ self.left, lhs @ self.right)

    @property
    def matrix(self):
        return self.left.matrix + self.right.matrix


def propagator_mul_gamma(propag: LatticePropagator, rhs: Gamma):
    if not isinstance(rhs, Gamma):
        return NotImplemented
    assert propag.latt_info.Ns == 4, "Ns should be 4"
    rhs = rhs.T  # S @ gamma = (gamma.T @ S.T).T
    result = LatticePropagator(propag.latt_info)
    for j in range(4):
        k = rhs.sparse_indices[j]
        gamma_kj = rhs.factor * rhs.sparse_data[j]
        result.data[:, :, :, :, :, :, j] = propag.data[:, :, :, :, :, :, k] * gamma_kj
    return result


def gamma_mul_propagator(lhs: Gamma, propag: LatticePropagator):
    if not isinstance(lhs, Gamma):
        return NotImplemented
    assert propag.latt_info.Ns == 4, "Ns should be 4"
    result = LatticePropagator(propag.latt_info)
    for i in range(4):
        k = lhs.sparse_indices[i]
        gamma_ik = lhs.factor * lhs.sparse_data[i]
        result.data[:, :, :, :, :, i, :] = gamma_ik * propag.data[:, :, :, :, :, k, :]
    return result


def gamma_mul_propagator_mul_gamma(lhs: Gamma, propag: LatticePropagator, rhs: Gamma):
    assert lhs is not None or rhs is not None, "lhs or rhs should not be None"
    if lhs is not None or rhs is not None:
        assert isinstance(lhs, Gamma) and isinstance(rhs, Gamma), "lhs and rhs should be Gamma"
        assert propag.latt_info.Ns == 4, "Ns should be 4"
        rhs = rhs.T  # S @ gamma = (gamma.T @ S.T).T
        result = LatticePropagator(propag.latt_info)
        for i in range(4):
            for j in range(4):
                k = lhs.sparse_indices[i]
                gamma_ik = lhs.factor * lhs.sparse_data[i]
                l = rhs.sparse_indices[j]  # noqa: E741
                gamma_lj = rhs.factor * rhs.sparse_data[j]
                result.data[:, :, :, :, :, i, j] = gamma_ik * gamma_lj * propag.data[:, :, :, :, :, k, l]
        return result
    elif lhs is None:
        return propagator_mul_gamma(propag, rhs)
    elif rhs is None:
        return gamma_mul_propagator(lhs, propag)


# Monkey patching for LatticePropagator
LatticePropagator.__matmul__ = lambda self, rhs: propagator_mul_gamma(self, rhs)
LatticePropagator.__rmatmul__ = lambda self, lhs: gamma_mul_propagator(lhs, self)
LatticePropagator.matmul = lambda self, lhs=None, rhs=None: gamma_mul_propagator_mul_gamma(lhs, self, rhs)


def gamma(n: int):
    assert isinstance(n, int) and 0 <= n <= 15, "n should be int from 0 to 15"
    return GammaMatrix.matrix(n)
