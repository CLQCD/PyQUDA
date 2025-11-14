from typing import List, Union

import numpy
from numpy.typing import NDArray

from pyquda_comm import getArrayBackend
from pyquda_comm.array import arrayDevice


class _GammaMatrix:
    gamma_0: NDArray[numpy.complex128]
    gamma_1: NDArray[numpy.complex128]
    gamma_2: NDArray[numpy.complex128]
    gamma_3: NDArray[numpy.complex128]
    gamma_4: NDArray[numpy.complex128]

    @classmethod
    def matrix(cls, index) -> NDArray[numpy.complex128]:
        gamma = (
            (cls.gamma_1 if index & 0b0001 else cls.gamma_0)
            @ (cls.gamma_2 if index & 0b0010 else cls.gamma_0)
            @ (cls.gamma_3 if index & 0b0100 else cls.gamma_0)
            @ (cls.gamma_4 if index & 0b1000 else cls.gamma_0)
        )
        return arrayDevice(gamma, getArrayBackend())


class DeGrandRossiMatrix(_GammaMatrix):
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


class DiracPauliMatrix(_GammaMatrix):
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
            [0, 0, 0, -1j],
            [0, 0, -1j, 0],
            [0, 1j, 0, 0],
            [1j, 0, 0, 0],
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
            [0, 0, -1j, 0],
            [0, 0, 0, 1j],
            [1j, 0, 0, 0],
            [0, -1j, 0, 0],
        ],
        "<c16",
    )
    gamma_4 = numpy.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ],
        "<c16",
    )


class _GammaSparse:
    gamma_indices: List[List[int]]
    gamma_data: List[List[Union[int, complex]]]

    @classmethod
    def indices(cls, index) -> List[int]:
        return cls.gamma_indices[index]

    @classmethod
    def data(cls, index) -> List[Union[int, complex]]:
        return cls.gamma_data[index]


class DeGrandRossiSparse(_GammaSparse):
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


class DiracPauliSparse(_GammaSparse):
    gamma_indices = [
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [0, 1, 2, 3],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [0, 1, 2, 3],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
    ]
    gamma_data = [
        [1, 1, 1, 1],
        [-1j, -1j, 1j, 1j],
        [-1, 1, 1, -1],
        [1j, -1j, 1j, -1j],
        [-1j, 1j, 1j, -1j],
        [-1, 1, -1, 1],
        [1j, 1j, 1j, 1j],
        [1, 1, -1, -1],
        [1, 1, -1, -1],
        [1j, 1j, 1j, 1j],
        [1, -1, 1, -1],
        [1j, -1j, -1j, 1j],
        [1j, -1j, 1j, -1j],
        [-1, 1, 1, -1],
        [1j, 1j, -1j, -1j],
        [-1, -1, -1, -1],
    ]


class Gamma:
    popcnt = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]
    popsign = [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1]

    def __init__(self, index: int, factor: Union[int, float, complex] = 1) -> None:
        assert isinstance(index, int) and 0b0000 <= index <= 0b1111, "index should be int from 0 to 15"
        self.index = index
        self.factor = factor

    def __repr__(self) -> str:
        factor = "-" if self.factor == -1 else "" if self.factor == 1 else f"{self.factor} * "
        return (
            f"{factor}"
            f"{'γ₀' if self.index == 0 else ''}"
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
        return self.factor * DeGrandRossiMatrix.matrix(self.index)

    @property
    def sparse_indices(self):
        return DeGrandRossiSparse.indices(self.index)

    @property
    def sparse_data(self):
        return [self.factor * value for value in DeGrandRossiSparse.data(self.index)]
