from typing import List, Literal

import numpy

from .pointer import ndarrayDataPointer


class LatticeInfo:
    Ns: int = 4
    Nc: int = 3
    Nd: int = 4

    def __init__(
        self,
        latt_size: List[int],
        t_boundary: Literal[1, -1] = -1,
        anisotropy: float = 1.0,
    ) -> None:
        from . import getMPIComm, getMPISize, getMPIRank, getGridSize, getGridCoord

        if getMPIComm() is None:
            raise RuntimeError("pyquda.init() must be called before contructing LatticeInfo")

        self.mpi_size = getMPISize()
        self.mpi_rank = getMPIRank()
        self.grid_size = getGridSize()
        self.grid_coord = getGridCoord()

        Gx, Gy, Gz, Gt = self.grid_size
        gx, gy, gz, gt = self.grid_coord
        Lx, Ly, Lz, Lt = latt_size

        assert (
            Lx % (2 * Gx) == 0 and Ly % (2 * Gy) == 0 and Lz % (2 * Gz) == 0 and Lt % (2 * Gt) == 0
        ), "Necessary for consistant even-odd preconditioning"
        self.Gx, self.Gy, self.Gz, self.Gt = Gx, Gy, Gz, Gt
        self.gx, self.gy, self.gz, self.gt = gx, gy, gz, gt
        self.global_size = [Lx, Ly, Lz, Lt]
        self.global_volume = Lx * Ly * Lz * Lt
        Lx, Ly, Lz, Lt = Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt
        self.Lx, self.Ly, self.Lz, self.Lt = Lx, Ly, Lz, Lt
        self.size = [Lx, Ly, Lz, Lt]
        self.volume = Lx * Ly * Lz * Lt
        self.size_cb2 = [Lx // 2, Ly, Lz, Lt]
        self.volume_cb2 = Lx * Ly * Lz * Lt // 2
        self.ga_pad = Lx * Ly * Lz * Lt // min(Lx, Ly, Lz, Lt) // 2

        self.t_boundary = t_boundary
        self.anisotropy = anisotropy


Ns, Nc, Nd = LatticeInfo.Ns, LatticeInfo.Nc, LatticeInfo.Nd


def lexico(data: numpy.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Np, Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    assert Np == 2, "There must be 2 parities."
    Lx *= 2
    Npre = int(numpy.prod(shape[: axes[0]]))
    Nsuf = int(numpy.prod(shape[axes[-1] + 1 :]))
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
    return data_lexico.reshape(*shape[: axes[0]], Lt, Lz, Ly, Lx, *shape[axes[-1] + 1 :])


def cb2(data: numpy.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    Npre = int(numpy.prod(shape[: axes[0]]))
    Nsuf = int(numpy.prod(shape[axes[-1] + 1 :]))
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
    return data_cb2.reshape(*shape[: axes[0]], 2, Lt, Lz, Ly, Lx // 2, *shape[axes[-1] + 1 :])


def newLatticeFieldData(latt_info: LatticeInfo, dtype: str):
    from . import getCUDABackend

    backend = getCUDABackend()
    Lx, Ly, Lz, Lt = latt_info.size
    if backend == "cupy":
        import cupy

        if dtype == "Gauge":
            ret = cupy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
            ret[:] = cupy.identity(Nc)
            return ret
        elif dtype == "Colorvector":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
        elif dtype == "Fermion":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c16")
        elif dtype == "Propagator":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), "<c16")
        elif dtype == "StaggeredFermion":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
        elif dtype == "StaggeredPropagator":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
        else:
            raise ValueError(f"Unsupported lattice field type {dtype}")
    elif backend == "torch":
        import torch

        if dtype == "Gauge":
            ret = torch.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), dtype=torch.complex128, device="cuda")
            ret[:] = torch.eye(Nc, device="cuda")
            return ret
        elif dtype == "Colorvector":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), dtype=torch.complex128, device="cuda")
        elif dtype == "Fermion":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), dtype=torch.complex128, device="cuda")
        elif dtype == "Propagator":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), dtype=torch.complex128, device="cuda")
        elif dtype == "StaggeredFermion":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), dtype=torch.complex128, device="cuda")
        elif dtype == "StaggeredPropagator":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Nc, Nc), dtype=torch.complex128, device="cuda")
        else:
            raise ValueError(f"Unsupported lattice field type {dtype}")
    else:
        raise ValueError(f"Unsupported CUDA backend {backend}")


class LatticeField:
    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info

    def backup(self):
        from . import getCUDABackend

        backend = getCUDABackend()
        if isinstance(self.data, numpy.ndarray):
            return self.data.copy()
        elif backend == "cupy":
            return self.data.copy()
        elif backend == "torch":
            return self.data.clone()
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")

    def toDevice(self):
        from . import getCUDABackend

        backend = getCUDABackend()
        if backend == "cupy":
            import cupy

            self.data = cupy.asarray(self.data)
        elif backend == "torch":
            import torch

            self.data = torch.as_tensor(self.data, device="cuda")
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")

    def toHost(self):
        from . import getCUDABackend

        backend = getCUDABackend()
        if isinstance(self.data, numpy.ndarray):
            pass
        elif backend == "cupy":
            self.data = self.data.get()
        elif backend == "torch":
            self.data = self.data.cpu().numpy()
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")

    def getHost(self):
        from . import getCUDABackend

        backend = getCUDABackend()
        if isinstance(self.data, numpy.ndarray):
            return self.data.copy()
        elif backend == "cupy":
            return self.data.get()
        elif backend == "torch":
            return self.data.cpu().numpy()
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")


class LatticeGauge(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.data = newLatticeFieldData(latt_info, "Gauge")
        else:
            self.data = value.reshape(Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc)
        self.pure_gauge = None

    def copy(self):
        return LatticeGauge(self.latt_info, self.backup())

    def setAntiPeroidicT(self):
        if self.latt_info.gt == self.latt_info.Gt - 1:
            self.data[Nd - 1, :, self.latt_info.Lt - 1] *= -1

    def setAnisotropy(self, anisotropy: float):
        self.data[: Nd - 1] /= anisotropy

    @property
    def data_ptr(self):
        return ndarrayDataPointer(self.data.reshape(-1), True)

    @property
    def data_ptrs(self):
        return ndarrayDataPointer(self.data.reshape(4, -1), True)

    def lexico(self):
        return lexico(self.getHost(), [1, 2, 3, 4, 5])

    def initPureGuage(self):
        if self.pure_gauge is None:
            from .dirac.pure_gauge import PureGauge

            self.pure_gauge = PureGauge(self.latt_info)

    def smearAPE(self, n_steps: int, alpha: float, dir: int):
        self.initPureGuage()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.smearAPE(n_steps, alpha, dir)
        self.pure_gauge.saveSmearedGauge(self)

    def smearSTOUT(self, n_steps: int, rho: float, dir: int):
        self.initPureGuage()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.smearSTOUT(n_steps, rho, dir)
        self.pure_gauge.saveSmearedGauge(self)

    def plaquette(self):
        self.initPureGuage()
        return self.pure_gauge.plaquette()

    def polyakovLoop(self):
        self.initPureGuage()
        return self.pure_gauge.polyakovLoop()

    def energy(self):
        self.initPureGuage()
        return self.pure_gauge.energy()

    def qcharge(self):
        self.initPureGuage()
        return self.pure_gauge.qcharge()


class LatticeFermion(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.data = newLatticeFieldData(latt_info, "Fermion")
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
        return lexico(self.getHost(), [0, 1, 2, 3, 4])


class LatticePropagator(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.data = newLatticeFieldData(latt_info, "Propagator")
        else:
            self.data = value.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc)

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])

    def transpose(self):
        return self.data.transpose(0, 1, 2, 3, 4, 6, 5, 8, 7).copy()


class LatticeStaggeredFermion(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.data = newLatticeFieldData(latt_info, "StaggeredFermion")
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
        return lexico(self.getHost(), [0, 1, 2, 3, 4])


class LatticeStaggeredPropagator(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.data = newLatticeFieldData(latt_info, "StaggeredPropagator")
        else:
            self.data = value.reshape(2, Lt, Lz, Ly, Lx // 2, Nc, Nc)

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])

    def transpose(self):
        return self.data.transpose(0, 1, 2, 3, 4, 6, 5).copy()
