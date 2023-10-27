from functools import lru_cache

from .. import getCUDABackend


class _ConstantCuPy:
    @staticmethod
    @lru_cache(1)
    def zero(cupy):
        return cupy.zeros((4, 4))

    @staticmethod
    @lru_cache(1)
    def one(cupy):
        return cupy.identity(4)

    @staticmethod
    @lru_cache(1)
    def gamma_0(cupy):
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
    def gamma_1(cupy):
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
    def gamma_2(cupy):
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
    def gamma_3(cupy):
        return cupy.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )


class _ConstantTorch:
    @staticmethod
    @lru_cache(1)
    def zero(torch):
        return torch.zeros((4, 4), device="cuda")

    @staticmethod
    @lru_cache(1)
    def one(torch):
        return torch.eye(4, device="cuda")

    @staticmethod
    @lru_cache(1)
    def gamma_0(torch):
        return torch.as_tensor(
            [
                [0, 0, 0, 1j],
                [0, 0, 1j, 0],
                [0, -1j, 0, 0],
                [-1j, 0, 0, 0],
            ],
            device="cuda",
        )

    @staticmethod
    @lru_cache(1)
    def gamma_1(torch):
        return torch.as_tensor(
            [
                [0, 0, 0, -1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
            ],
            device="cuda",
        )

    @staticmethod
    @lru_cache(1)
    def gamma_2(torch):
        return torch.as_tensor(
            [
                [0, 0, 1j, 0],
                [0, 0, 0, -1j],
                [-1j, 0, 0, 0],
                [0, 1j, 0, 0],
            ],
            device="cuda",
        )

    @staticmethod
    @lru_cache(1)
    def gamma_3(torch):
        return torch.as_tensor(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            device="cuda",
        )


def gamma(n: int):
    assert isinstance(n, int) and n >= 0 and n <= 15
    backend = getCUDABackend()
    if backend == "cupy":
        import cupy

        return cupy.asarray(
            (_ConstantCuPy.gamma_0(cupy) if n & 0b0001 else _ConstantCuPy.one(cupy))
            @ (_ConstantCuPy.gamma_1(cupy) if n & 0b0010 else _ConstantCuPy.one(cupy))
            @ (_ConstantCuPy.gamma_2(cupy) if n & 0b0100 else _ConstantCuPy.one(cupy))
            @ (_ConstantCuPy.gamma_3(cupy) if n & 0b1000 else _ConstantCuPy.one(cupy))
        )
    elif backend == "torch":
        import torch

        return torch.as_tensor(
            (_ConstantTorch.gamma_0(torch) if n & 0b0001 else _ConstantTorch.one(torch))
            @ (_ConstantTorch.gamma_1(torch) if n & 0b0010 else _ConstantTorch.one(torch))
            @ (_ConstantTorch.gamma_2(torch) if n & 0b0100 else _ConstantTorch.one(torch))
            @ (_ConstantTorch.gamma_3(torch) if n & 0b1000 else _ConstantTorch.one(torch)),
            device="cuda",
        )


def bilateral_apply(data, out, axis, gamma_left, gamma_right, conj):
    backend = getCUDABackend()
    if backend == "torch":
        raise ValueError("`bilateral_apply` doesn't support torch backend yet.")
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
