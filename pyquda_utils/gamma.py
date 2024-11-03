from copy import deepcopy
from typing import List, Literal, NamedTuple, Union

import numpy

from pyquda import getCUDABackend, getLogger


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
    def matrix(cls, index) -> numpy.ndarray:
        return (
            (cls.gamma_1 if index & 0b0001 else cls.gamma_0)
            @ (cls.gamma_2 if index & 0b0010 else cls.gamma_0)
            @ (cls.gamma_3 if index & 0b0100 else cls.gamma_0)
            @ (cls.gamma_4 if index & 0b1000 else cls.gamma_0)
        )


class GammaCsrMatrix(NamedTuple):
    indices: List[int]
    data: List[Union[int, complex]]


class GammaSparse:
    indices = [
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
    data = [
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
    def csr_matrix(cls, index) -> GammaCsrMatrix:
        return GammaCsrMatrix(deepcopy(cls.indices[index]), deepcopy(cls.data[index]))


class Gamma:
    popcnt = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]
    popsign = [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1]

    def __init__(self, index: int, sign: Literal[1, -1]) -> None:
        assert isinstance(index, int) and 0b0000 <= index <= 0b1111, "index should be int from 0 to 15"
        self.index = index
        self.sign = sign

    def __repr__(self) -> str:
        return (
            f"{'-' if self.sign == -1 else ''}"
            f"{'γ₀' if not self.index else ''}"
            f"{'γ₁' if self.index & 0b0001 else ''}"
            f"{'γ₂' if self.index & 0b0010 else ''}"
            f"{'γ₃' if self.index & 0b0100 else ''}"
            f"{'γ₄' if self.index & 0b1000 else ''}"
        )

    def __matmul__(self, rhs: "Gamma") -> "Gamma":
        index = self.index ^ rhs.index
        sign = self.sign * rhs.sign
        if self.index & 0b1000:
            sign *= Gamma.popsign[rhs.index & 0b0111]
        if self.index & 0b0100:
            sign *= Gamma.popsign[rhs.index & 0b0011]
        if self.index & 0b0010:
            sign *= Gamma.popsign[rhs.index & 0b0001]
        # if self.index & 0b0001:
        #     sign *= Gamma.popsign[rhs.index & 0b0000]
        return Gamma(index, sign)

    @property
    def dagger(self) -> "Gamma":
        index = self.index
        sign = self.sign * Gamma.popsign[self.index]
        return Gamma(index, sign)

    @property
    def matrix(self) -> numpy.ndarray:
        return self.sign * GammaMatrix.matrix(self.index)

    @property
    def csr_matrix(self):
        return GammaSparse.csr_matrix(self.index)


def gamma(n: int):
    assert isinstance(n, int) and 0 <= n <= 15, "n should be int from 0 to 15"
    backend = getCUDABackend()
    if backend == "numpy":
        return GammaMatrix.matrix(n)
    elif backend == "cupy":
        import cupy

        return cupy.asarray(GammaMatrix.matrix(n))
    elif backend == "torch":
        import torch

        return torch.as_tensor(GammaMatrix.matrix(n))


def bilateral_apply(data, out, axis, gamma_left, gamma_right, conj):
    backend = getCUDABackend()
    if backend != "cupy":
        getLogger().critical(f"`bilateral_apply` doesn't support {backend} backend yet", ValueError)
    import cupy

    gamma_left = cupy.sparse.csr_matrix(gamma_left)
    gamma_right = cupy.sparse.csc_matrix(gamma_right)
    shape = data.shape
    assert (
        axis[1] - axis[0] == 1 and shape[axis[0]] == 4 and shape[axis[1]] == 4
    ), "Indices for Ns must be continuous and Ns must be 4"
    p = 1
    for i in range(axis[0]):
        p *= shape[i]
    data = data.reshape(p, 4, 4, -1)
    out = out.reshape(p, 4, 4, -1)
    for i in range(4):
        for j in range(4):
            ii = gamma_left.indices[i]
            jj = gamma_right.indices[j]
            if conj:
                out[:, j, i] = (gamma_left.data[i] * gamma_right.data[j]) * data[:, jj, ii].conj()
            else:
                out[:, i, j] = (gamma_left.data[i] * gamma_right.data[j]) * data[:, ii, jj]
