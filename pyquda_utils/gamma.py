from pyquda_comm.field import LatticeFermion, LatticePropagator
from pyquda.gamma import DeGrandRossiMatrix, Gamma, Projector

__all__ = ["gamma", "Gamma", "Projector"]


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
