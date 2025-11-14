from typing import Union

from pyquda_comm.field import LatticeFermion, LatticePropagator
from pyquda.gamma import DeGrandRossiMatrix, Gamma


def gamma_add_gamma(lhs: Gamma, rhs: Gamma) -> Union[Gamma, "Polarize"]:
    if not isinstance(rhs, Gamma):
        return NotImplemented
    if lhs.index == rhs.index:
        return Gamma(lhs.index, lhs.factor + rhs.factor)
    else:
        return Polarize(lhs, rhs)


def gamma_sub_gamma(lhs: Gamma, rhs: Gamma) -> Union[Gamma, "Polarize"]:
    if not isinstance(rhs, Gamma):
        return NotImplemented
    if lhs.index == rhs.index:
        return Gamma(lhs.index, lhs.factor - rhs.factor)
    else:
        return Polarize(lhs, -rhs)


Gamma.__add__ = lambda self, rhs: gamma_add_gamma(self, rhs)
Gamma.__sub__ = lambda self, rhs: gamma_sub_gamma(self, rhs)


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
            if not isinstance(self.right.factor, complex) and self.right.factor > 0
            else f"{self.left} - {Gamma(self.right.index, -self.right.factor)}"
        )

    def __neg__(self) -> "Polarize":
        return Polarize(-self.left, -self.right)

    def __mul__(self, rhs: Union[int, float, complex]):
        if not isinstance(rhs, (int, float, complex)):
            return NotImplemented
        return Polarize(self.left * rhs, self.right * rhs)

    def __rmul__(self, lhs: Union[int, float, complex]):
        if not isinstance(lhs, (int, float, complex)):
            return NotImplemented
        return Polarize(lhs * self.left, lhs * self.right)

    def __truediv__(self, rhs: Union[int, float, complex]):
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


def gamma_mul_fermion(lhs: Gamma, propag: LatticeFermion):
    if not isinstance(lhs, Gamma):
        return NotImplemented
    assert propag.latt_info.Ns == 4, "Ns should be 4"
    result = LatticeFermion(propag.latt_info)
    for i in range(4):
        k = lhs.sparse_indices[i]
        gamma_ik = lhs.factor * lhs.sparse_data[i]
        result.data[:, :, :, :, :, i] = gamma_ik * propag.data[:, :, :, :, :, k]
    return result


def propagator_mul_gamma(lhs: LatticePropagator, rhs: Gamma):
    if not isinstance(rhs, Gamma):
        return NotImplemented
    assert lhs.latt_info.Ns == 4, "Ns should be 4"
    rhs = rhs.T  # S @ gamma = (gamma.T @ S.T).T
    result = LatticePropagator(lhs.latt_info)
    for j in range(4):
        k = rhs.sparse_indices[j]
        gamma_kj = rhs.factor * rhs.sparse_data[j]
        result.data[:, :, :, :, :, :, j] = lhs.data[:, :, :, :, :, :, k] * gamma_kj
    return result


def gamma_mul_propagator(lhs: Gamma, rhs: LatticePropagator):
    if not isinstance(lhs, Gamma):
        return NotImplemented
    assert rhs.latt_info.Ns == 4, "Ns should be 4"
    result = LatticePropagator(rhs.latt_info)
    for i in range(4):
        k = lhs.sparse_indices[i]
        gamma_ik = lhs.factor * lhs.sparse_data[i]
        result.data[:, :, :, :, :, i, :] = gamma_ik * rhs.data[:, :, :, :, :, k, :]
    return result


def gamma_mul_propagator_mul_gamma(lhs: Gamma, propag: LatticePropagator, rhs: Gamma):
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


# Monkey patching for LatticePropagator
LatticeFermion.__rmatmul__ = lambda self, lhs: gamma_mul_fermion(self, lhs)
LatticePropagator.__matmul__ = lambda self, rhs: propagator_mul_gamma(self, rhs)
LatticePropagator.__rmatmul__ = lambda self, lhs: gamma_mul_propagator(lhs, self)
LatticePropagator.matmul = lambda self, lhs, rhs: gamma_mul_propagator_mul_gamma(lhs, self, rhs)


def gamma(n: int):
    assert isinstance(n, int) and 0 <= n <= 15, "n should be int from 0 to 15"
    return DeGrandRossiMatrix.matrix(n)
