from functools import lru_cache

import cupy


class _Constant:
    @staticmethod
    @lru_cache(1)
    def zero():
        return cupy.zeros((4, 4))

    @staticmethod
    @lru_cache(1)
    def one():
        return cupy.identity(4)

    @staticmethod
    @lru_cache(1)
    def gamma_0():
        return cupy.array(
            [
                [0, 0, 0, 1j],
                [0, 0, 1j, 0],
                [0, -1j, 0, 0],
                [-1j, 0, 0, 0],
            ]
        )

    @staticmethod
    @lru_cache(1)
    def gamma_1():
        return cupy.array(
            [
                [0, 0, 0, -1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
            ]
        )

    @staticmethod
    @lru_cache(1)
    def gamma_2():
        return cupy.array(
            [
                [0, 0, 1j, 0],
                [0, 0, 0, -1j],
                [-1j, 0, 0, 0],
                [0, 1j, 0, 0],
            ]
        )

    @staticmethod
    @lru_cache(1)
    def gamma_3():
        return cupy.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )


def gamma(n: int):
    assert isinstance(n, int) and n >= 0 and n <= 15
    return cupy.asarray(
        (_Constant.gamma_0() if n & 0b0001 else _Constant.one())
        @ (_Constant.gamma_1() if n & 0b0010 else _Constant.one())
        @ (_Constant.gamma_2() if n & 0b0100 else _Constant.one())
        @ (_Constant.gamma_3() if n & 0b1000 else _Constant.one())
    )


def bilateral_apply(data, out, axis, gamma_left, gamma_right, conj):
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
