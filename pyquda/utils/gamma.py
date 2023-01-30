from functools import lru_cache

import cupy as cp


class _Constant:
    @staticmethod
    @lru_cache(1)
    def zero():
        return cp.zeros((4, 4))

    @staticmethod
    @lru_cache(1)
    def one():
        return cp.identity(4)

    @staticmethod
    @lru_cache(1)
    def gamma_0():
        return cp.array([
            [0, 0, 0, 1j],
            [0, 0, 1j, 0],
            [0, -1j, 0, 0],
            [-1j, 0, 0, 0],
        ])

    @staticmethod
    @lru_cache(1)
    def gamma_1():
        return cp.array([
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
        ])

    @staticmethod
    @lru_cache(1)
    def gamma_2():
        return cp.array([
            [0, 0, 1j, 0],
            [0, 0, 0, -1j],
            [-1j, 0, 0, 0],
            [0, 1j, 0, 0],
        ])

    @staticmethod
    @lru_cache(1)
    def gamma_3():
        return cp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])


def gamma(n: int):
    assert isinstance(n, int) and n >= 0 and n <= 15
    return cp.asarray(
        (_Constant.gamma_0() if n & 0b0001 else _Constant.one())
        @ (_Constant.gamma_1() if n & 0b0010 else _Constant.one())
        @ (_Constant.gamma_2() if n & 0b0100 else _Constant.one())
        @ (_Constant.gamma_3() if n & 0b1000 else _Constant.one())
    )


def bilateral_apply(data, out, axis, gamma_left, gamma_right, conj):
    gamma_left = cp.sparse.csr_matrix(gamma_left)
    gamma_right = cp.sparse.csc_matrix(gamma_right)
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


_naming_scheme = {
    "a0": [[0]],
    "pi": [[15]],
    "pi(2)": [[8, 15]],
    "b0": [[8]],
    "rho": [[1], [2], [4]],
    "rho(2)": [[8, 1], [8, 2], [8, 4]],
    "a1": [[15, 1], [15, 2], [15, 4]],
    "b1": [[8, 15, 1], [8, 15, 2], [8, 15, 4]],
}

_naming_hermition = {
    "a0": "+",
    "pi": "-",
    "pi(2)": "+",
    "b0": "+",
    "rho": "-",
    "rho(2)": "+",
    "a1": "-",
    "b1": "-",
}

_naming_parity = {
    "a0": "+",
    "pi": "-",
    "pi(2)": "-",
    "b0": "+",
    "rho": "-",
    "rho(2)": "-",
    "a1": "+",
    "b1": "+",
}

_naming_charge_parity = {
    "a0": "+",
    "pi": "+",
    "pi(2)": "+",
    "b0": "-",
    "rho": "-",
    "rho(2)": "-",
    "a1": "+",
    "b1": "-",
}


def scheme(name: str):
    assert name in _naming_scheme
    return _naming_scheme[name]


def hermition(name: str):
    return _naming_hermition[name]


class GAMMA_NAME:
    A0 = "a0"
    PI = "pi"
    PI_2 = "pi(2)"
    B0 = "b0"
    RHO = "rho"
    RHO_2 = "rho(2)"
    A1 = "a1"
    B1 = "b1"


def instance(gamma_idxs: list):
    ret = _Constant.one()
    for idx in gamma_idxs:
        ret = ret @ gamma(idx)
    return ret
