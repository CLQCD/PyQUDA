from typing import List, Literal, Union

import numpy

from .pointer import ndarrayPointer, Pointer, Pointers


class LatticeInfo:
    Ns: int = 4
    Nc: int = 3
    Nd: int = 4

    def __init__(self, latt_size: List[int], t_boundary: Literal[1, -1] = 1, anisotropy: float = 1.0) -> None:
        self._checkLattice(latt_size)
        self._setLattice(latt_size, t_boundary, anisotropy)

    def _checkLattice(self, latt_size: List[int]):
        from . import getLogger, getGridSize

        if getGridSize() is None:
            getLogger().critical("pyquda.init() must be called before contructing LatticeInfo", RuntimeError)
        Gx, Gy, Gz, Gt = getGridSize()
        Lx, Ly, Lz, Lt = latt_size
        if not (Lx % (2 * Gx) == 0 and Ly % (2 * Gy) == 0 and Lz % (2 * Gz) == 0 and Lt % (2 * Gt) == 0):
            getLogger().critical(
                "lattice size must be divisible by gird size, "
                "and sublattice size must be even in every direction for consistant even-odd preconditioning",
                ValueError,
            )

    def _setLattice(self, latt_size: List[int], t_boundary: Literal[1, -1], anisotropy: float):
        from . import getMPIComm, getMPISize, getMPIRank, getGridSize, getGridCoord

        self.mpi_comm = getMPIComm()
        self.mpi_size = getMPISize()
        self.mpi_rank = getMPIRank()
        self.grid_size = getGridSize()
        self.grid_coord = getGridCoord()

        Gx, Gy, Gz, Gt = self.grid_size
        gx, gy, gz, gt = self.grid_coord
        Lx, Ly, Lz, Lt = latt_size

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


class LaplaceLatticeInfo(LatticeInfo):
    def __init__(self, latt_size: List[int]):
        self._checkLatticeOddT(latt_size)
        self._setLattice(latt_size, 1, 1.0)

    def _checkLatticeOddT(self, latt_size: List[int]):
        from . import getLogger, getGridSize

        if getGridSize() is None:
            getLogger().critical("pyquda.init() must be called before contructing LatticeInfo", RuntimeError)
        Gx, Gy, Gz, Gt = getGridSize()
        Lx, Ly, Lz, Lt = latt_size
        if not (Lx % (2 * Gx) == 0 and Ly % (2 * Gy) == 0 and Lz % (2 * Gz) == 0 and Lt % Gt == 0):
            getLogger().critical(
                "lattice size must be divisible by gird size, "
                "and sublattice size must be even in spacial direction for consistant even-odd preconditioning",
                ValueError,
            )


Ns, Nc, Nd = LatticeInfo.Ns, LatticeInfo.Nc, LatticeInfo.Nd


def lexico(data: numpy.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Np, Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    assert Np == 2
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


def newLatticeFieldData(latt_info: LatticeInfo, field: str):
    from . import getCUDABackend

    backend = getCUDABackend()
    Lx, Ly, Lz, Lt = latt_info.size
    if backend == "numpy":
        if field == "Gauge":
            ret = numpy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
            ret[:] = numpy.identity(Nc)
            return ret
        elif field == "Fermion":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c16")
        elif field == "Propagator":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), "<c16")
        elif field == "StaggeredFermion":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
        elif field == "StaggeredPropagator":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
        elif field == "Clover":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, 2, ((Ns // 2) * Nc) ** 2), "<f8")
        elif field == "Mom":
            return numpy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, 10), "<f8")
    elif backend == "cupy":
        import cupy

        if field == "Gauge":
            ret = cupy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
            ret[:] = cupy.identity(Nc)
            return ret
        elif field == "Fermion":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c16")
        elif field == "Propagator":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), "<c16")
        elif field == "StaggeredFermion":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
        elif field == "StaggeredPropagator":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
        elif field == "Clover":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, 2, ((Ns // 2) * Nc) ** 2), "<f8")
        elif field == "Mom":
            return cupy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, 10), "<f8")
    elif backend == "torch":
        import torch

        if field == "Gauge":
            ret = torch.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), dtype=torch.complex128)
            ret[:] = torch.eye(Nc)
            return ret
        elif field == "Fermion":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), dtype=torch.complex128)
        elif field == "Propagator":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), dtype=torch.complex128)
        elif field == "StaggeredFermion":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), dtype=torch.complex128)
        elif field == "StaggeredPropagator":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Nc, Nc), dtype=torch.complex128)
        elif field == "Clover":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, 2, ((Ns // 2) * Nc) ** 2), dtype=torch.float64)
        elif field == "Mom":
            return torch.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, 10), dtype=torch.float64)


def newMultiLatticeFieldData(latt_info: LatticeInfo, L5: int, field: str):
    from . import getCUDABackend

    backend = getCUDABackend()
    Lx, Ly, Lz, Lt = latt_info.size
    if backend == "numpy":
        if field == "Fermion":
            return numpy.zeros((L5, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c16")
        elif field == "StaggeredFermion":
            return numpy.zeros((L5, 2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
    elif backend == "cupy":
        import cupy

        if field == "Fermion":
            return cupy.zeros((L5, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c16")
        elif field == "StaggeredFermion":
            return cupy.zeros((L5, 2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
    elif backend == "torch":
        import torch

        if field == "Fermion":
            return torch.zeros((L5, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc), dtype=torch.complex128)
        elif field == "StaggeredFermion":
            return torch.zeros((L5, 2, Lt, Lz, Ly, Lx // 2, Nc), dtype=torch.complex128)


class LatticeField:
    def __init__(self, latt_info: LatticeInfo) -> None:
        from . import getCUDABackend

        self.latt_info = latt_info
        self._data = None
        self.backend = getCUDABackend()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, numpy.ndarray) or self.backend == "numpy":
            self._data = numpy.ascontiguousarray(value)
        elif self.backend == "cupy":
            import cupy

            self._data = cupy.ascontiguousarray(value)
        elif self.backend == "torch":
            self._data = value.contiguous()

    @property
    def location(self) -> Literal["numpy", "cupy", "torch"]:
        if isinstance(self.data, numpy.ndarray):
            return "numpy"
        else:
            return self.backend

    def setData(self, data):
        self.data = data

    def backup(self):
        location = self.location
        if location == "numpy":
            return self.data.copy()
        elif location == "cupy":
            return self.data.copy()
        elif location == "torch":
            return self.data.clone()

    def copy(self):
        return self.__class__(self.latt_info, self.backup())

    def toDevice(self):
        backend = self.backend
        if backend == "numpy":
            pass
        elif backend == "cupy":
            import cupy

            self.data = cupy.asarray(self.data)
        elif backend == "torch":
            import torch

            self.data = torch.as_tensor(self.data)

    def toHost(self):
        location = self.location
        if location == "numpy":
            pass
        elif location == "cupy":
            self.data = self.data.get()
        elif location == "torch":
            self.data = self.data.cpu().numpy()

    def getHost(self):
        location = self.location
        if location == "numpy":
            return self.data.copy()
        elif location == "cupy":
            return self.data.get()
        elif location == "torch":
            return self.data.cpu().numpy()

    def __add__(self, other):
        assert self.__class__ == other.__class__ and self.location == other.location
        return self.__class__(self.latt_info, self.data + other.data)

    def __sub__(self, other):
        assert self.__class__ == other.__class__ and self.location == other.location
        return self.__class__(self.latt_info, self.data - other.data)

    def __mul__(self, other):
        return self.__class__(self.latt_info, self.data * other)

    def __lmul__(self, other):
        return self.__class__(self.latt_info, other * self.data)

    def __truediv__(self, other):
        return self.__class__(self.latt_info, self.data / other)

    def __iadd__(self, other):
        assert self.__class__ == other.__class__ and self.location == other.location
        self._data += other.data
        return self

    def __isub__(self, other):
        assert self.__class__ == other.__class__ and self.location == other.location
        self._data -= other.data
        return self

    def __imul__(self, other):
        self._data *= other
        return self

    def __itruediv__(self, other):
        self._data /= other
        return self


class MultiLatticeField(LatticeField):
    def __init__(self, latt_info: LatticeInfo, L5: int) -> None:
        super().__init__(latt_info)
        self.L5 = L5

    def copy(self):
        return self.__class__(self.latt_info, self.L5, self.backup())

    def __add__(self, other):
        assert self.__class__ == other.__class__ and self.location == other.location
        return self.__class__(self.latt_info, self.L5, self.data + other.data)

    def __sub__(self, other):
        assert self.__class__ == other.__class__ and self.location == other.location
        return self.__class__(self.latt_info, self.L5, self.data - other.data)

    def __mul__(self, other):
        return self.__class__(self.latt_info, self.L5, self.data * other)

    def __lmul__(self, other):
        return self.__class__(self.latt_info, self.L5, other * self.data)

    def __truediv__(self, other):
        return self.__class__(self.latt_info, self.L5, self.data / other)


class LatticeGauge(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "Gauge"))
        else:
            self.setData(value.reshape(Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc))
        self.pure_gauge = None

    def setAntiPeriodicT(self):
        if self.latt_info.gt == self.latt_info.Gt - 1:
            self.data[Nd - 1, :, self.latt_info.Lt - 1] *= -1

    def setAnisotropy(self, anisotropy: float):
        self.data[: Nd - 1] /= anisotropy

    @property
    def data_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)

    @property
    def data_ptrs(self) -> Pointers:
        return ndarrayPointer(self.data.reshape(4, -1), True)

    def lexico(self):
        return lexico(self.getHost(), [1, 2, 3, 4, 5])

    def ensurePureGauge(self):
        if self.pure_gauge is None:
            from .dirac.pure_gauge import PureGauge

            self.pure_gauge = PureGauge(self.latt_info)

    def covDev(self, x: "LatticeFermion", covdev_mu: int) -> "LatticeFermion":
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        b = self.pure_gauge.covDev(x, covdev_mu)
        self.pure_gauge.freeGauge()
        return b

    def laplace(self, x: "LatticeStaggeredFermion", laplace3D: int) -> "LatticeStaggeredFermion":
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        b = self.pure_gauge.laplace(x, laplace3D)
        self.pure_gauge.freeGauge()
        return b

    def wuppertalSmear(self, x: Union["LatticeFermion", "LatticeStaggeredFermion"], n_steps: int, alpha: float):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        b = self.pure_gauge.wuppertalSmear(x, n_steps, alpha)
        self.pure_gauge.freeGauge()
        return b

    def staggeredPhase(self):
        self.ensurePureGauge()
        self.pure_gauge.staggeredPhase(self)

    def projectSU3(self, tol: float):
        self.ensurePureGauge()
        self.pure_gauge.projectSU3(self, tol)

    def path(self, paths: List[List[int]]):
        self.ensurePureGauge()
        return self.pure_gauge.path(self, paths)

    def loop(self, loops: List[List[List[int]]], coeff: List[float]):
        self.ensurePureGauge()
        return self.pure_gauge.loop(self, loops, coeff)

    def loopTrace(self, loops: List[List[int]]):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        traces = self.pure_gauge.loopTrace(loops)
        self.pure_gauge.freeGauge()
        return traces

    def apeSmear(self, n_steps: int, alpha: float, dir_ignore: int):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.apeSmear(n_steps, alpha, dir_ignore)
        self.pure_gauge.saveSmearedGauge(self)
        self.pure_gauge.freeGauge()
        self.pure_gauge.freeSmearedGauge()

    def stoutSmear(self, n_steps: int, rho: float, dir_ignore: int):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.stoutSmear(n_steps, rho, dir_ignore)
        self.pure_gauge.saveSmearedGauge(self)
        self.pure_gauge.freeGauge()
        self.pure_gauge.freeSmearedGauge()

    def hypSmear(self, n_steps: int, alpha1: float, alpha2: float, alpha3: float, dir_ignore: int):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore)
        self.pure_gauge.saveSmearedGauge(self)
        self.pure_gauge.freeGauge()
        self.pure_gauge.freeSmearedGauge()

    def wilsonFlow(self, n_steps: int, epsilon: float):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.wilsonFlow(1, epsilon, 0, False)
        energy = [self.pure_gauge.obs_param.energy]
        for step in range(1, n_steps):
            self.pure_gauge.wilsonFlow(1, epsilon, step * epsilon, True)
            energy.append(self.pure_gauge.obs_param.energy)
        self.pure_gauge.saveSmearedGauge(self)  # Save before the last step
        self.pure_gauge.wilsonFlow(1, epsilon, n_steps * epsilon, True)
        energy.append(self.pure_gauge.obs_param.energy)
        self.pure_gauge.freeGauge()
        self.pure_gauge.freeSmearedGauge()
        return energy

    def wilsonFlowScale(self, max_steps: int, epsilon: float):
        from . import getLogger

        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.wilsonFlow(1, epsilon, 0, False)
        t2E, tdt2E = 0, 0
        t0, w0 = 0, 0
        for step in range(1, max_steps + 1):
            if t2E >= 0.3 and tdt2E >= 0.3:
                break
            self.pure_gauge.wilsonFlow(1, epsilon, step * epsilon, True)
            t2E_old, t2E = t2E, (step * epsilon) ** 2 * self.pure_gauge.obs_param.energy[0]
            tdt2E_old, tdt2E = tdt2E, (step - 0.5) * (t2E - t2E_old)
            if t0 == 0 and t2E >= 0.3:
                t0 = (step - (t2E - 0.3) / (t2E - t2E_old)) * epsilon
            if w0 == 0 and tdt2E >= 0.3:
                w0 = ((step - 0.5 - (tdt2E - 0.3) / (tdt2E - tdt2E_old)) * epsilon) ** 0.5
            getLogger().info(f"t2E({step * epsilon})={t2E}, tdt2E({(step - 0.5) * epsilon})={tdt2E}")
        else:
            getLogger().error(
                f"Wilson flow scale doesn't exceed 0.3 at max_steps*epsilon={max_steps*epsilon}", RuntimeError
            )
        self.pure_gauge.freeGauge()
        self.pure_gauge.freeSmearedGauge()
        return t0, w0

    def symanzikFlow(self, n_steps: int, epsilon: float):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.symanzikFlow(1, epsilon, 0, False)
        energy = [self.pure_gauge.obs_param.energy]
        for step in range(1, n_steps):
            self.pure_gauge.symanzikFlow(1, epsilon, step * epsilon, True)
            energy.append(self.pure_gauge.obs_param.energy)
        self.pure_gauge.saveSmearedGauge(self)  # Save before the last step
        self.pure_gauge.symanzikFlow(1, epsilon, n_steps * epsilon, True)
        energy.append(self.pure_gauge.obs_param.energy)
        self.pure_gauge.freeGauge()
        self.pure_gauge.freeSmearedGauge()
        return energy

    def symanzikFlowScale(self, max_steps: int, epsilon: float):
        from . import getLogger

        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.symanzikFlow(1, epsilon, 0, False)
        t2E, tdt2E = 0, 0
        t0, w0 = 0, 0
        for step in range(1, max_steps + 1):
            if t2E >= 0.3 and tdt2E >= 0.3:
                break
            self.pure_gauge.symanzikFlow(1, epsilon, step * epsilon, True)
            t2E_old, t2E = t2E, (step * epsilon) ** 2 * self.pure_gauge.obs_param.energy[0]
            tdt2E_old, tdt2E = tdt2E, (step - 0.5) * (t2E - t2E_old)
            if t0 == 0 and t2E >= 0.3:
                t0 = (step - (t2E - 0.3) / (t2E - t2E_old)) * epsilon
            if w0 == 0 and tdt2E >= 0.3:
                w0 = ((step - 0.5 - (tdt2E - 0.3) / (tdt2E - tdt2E_old)) * epsilon) ** 0.5
            getLogger().info(f"t2E({step * epsilon})={t2E}, tdt2E({(step - 0.5) * epsilon})={tdt2E}")
        else:
            getLogger().error(
                f"Symanzik flow scale doesn't exceed 0.3 at max_steps*epsilon={max_steps*epsilon}", RuntimeError
            )
        self.pure_gauge.freeGauge()
        self.pure_gauge.freeSmearedGauge()
        return t0, w0

    def smearAPE(self, n_steps: int, factor: float, dir_ignore: int):
        """A variant of apeSmear() to match Chroma"""
        dimAPE = 3 if dir_ignore >= 0 and dir_ignore <= 3 else 4
        self.apeSmear(n_steps, (dimAPE - 1) / (dimAPE - 1 + factor / 2), dir_ignore)

    def smearSTOUT(self, n_steps: int, rho: float, dir_ignore: int):
        self.stoutSmear(n_steps, rho, dir_ignore)

    def smearHYP(self, n_steps: int, alpha1: float, alpha2: float, alpha3: float, dir_ignore: int):
        self.hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore)

    def flowWilson(self, n_steps: int, time: float):
        return self.wilsonFlow(n_steps, time / n_steps)

    def flowWilsonScale(self, epsilon: float):
        return self.wilsonFlowScale(100000, epsilon)

    def flowSymanzik(self, n_steps: int, time: float):
        return self.symanzikFlow(n_steps, time / n_steps)

    def flowSymanzikScale(self, epsilon: float):
        return self.symanzikFlowScale(100000, epsilon)

    def plaquette(self):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        plaquette = self.pure_gauge.plaquette()
        self.pure_gauge.freeGauge()
        return plaquette

    def polyakovLoop(self):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        polyakovLoop = self.pure_gauge.polyakovLoop()
        self.pure_gauge.freeGauge()
        return polyakovLoop

    def energy(self):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        energy = self.pure_gauge.energy()
        self.pure_gauge.freeGauge()
        return energy

    def qcharge(self):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        qcharge = self.pure_gauge.qcharge()
        self.pure_gauge.freeGauge()
        return qcharge

    def qchargeDensity(self):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        qcharge_density = self.pure_gauge.qchargeDensity()
        self.pure_gauge.freeGauge()
        return qcharge_density

    def gauss(self, seed: int, sigma: float):
        """
        Generate Gaussian distributed fields and store in the
        resident gauge field.  We create a Gaussian-distributed su(n)
        field and exponentiate it, e.g., U = exp(sigma * H), where H is
        the distributed su(n) field and sigma is the width of the
        distribution (sigma = 0 results in a free field, and sigma = 1 has
        maximum disorder).

        seed: int
            The seed used for the RNG
        sigma: float
            Width of Gaussian distrubution
        """
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.gaussGauge(seed, sigma)
        self.pure_gauge.saveGauge(self)
        self.pure_gauge.freeGauge()

    def fixingOVR(
        self,
        gauge_dir: Literal[3, 4],
        Nsteps: int,
        verbose_interval: int,
        relax_boost: float,
        tolerance: float,
        reunit_interval: int,
        stopWtheta: int,
    ):
        """
        Gauge fixing with overrelaxation with support for single and multi GPU.

        Parameters
        ----------
        gauge_dir: {3, 4}
            3 for Coulomb gauge fixing, 4 for Landau gauge fixing
        Nsteps: int
            maximum number of steps to perform gauge fixing
        verbose_interval: int
            print gauge fixing info when iteration count is a multiple of this
        relax_boost: float
            gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
        tolerance: float
            torelance value to stop the method, if this value is zero then the method stops when
            iteration reachs the maximum number of steps defined by Nsteps
        reunit_interval: int
            reunitarize gauge field when iteration count is a multiple of this
        stopWtheta: int
            0 for MILC criterion and 1 to use the theta value
        """
        self.ensurePureGauge()
        self.pure_gauge.fixingOVR(
            self, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta
        )

    def fixingFFT(
        self,
        gauge_dir: Literal[3, 4],
        Nsteps: int,
        verbose_interval: int,
        alpha: float,
        autotune: int,
        tolerance: float,
        stopWtheta: int,
    ):
        """
        Gauge fixing with Steepest descent method with FFTs with support for single GPU only.

        Parameters
        ----------
        gauge_dir: {3, 4}
            3 for Coulomb gauge fixing, 4 for Landau gauge fixing
        Nsteps: int
            maximum number of steps to perform gauge fixing
        verbose_interval: int
            print gauge fixing info when iteration count is a multiple of this
        alpha: float
            gauge fixing parameter of the method, most common value is 0.08
        autotune: int
            1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
        tolerance: float
            torelance value to stop the method, if this value is zero then the method stops when
            iteration reachs the maximum number of steps defined by Nsteps
        stopWtheta: int
            0 for MILC criterion and 1 to use the theta value
        """
        self.ensurePureGauge()
        self.pure_gauge.fixingFFT(self, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta)


class LatticeMom(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "Mom"))
        else:
            self.setData(value.reshape(Nd, 2, Lt, Lz, Ly, Lx // 2, 10))
        self.pure_gauge = None

    @property
    def data_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)

    @property
    def data_ptrs(self) -> Pointers:
        return ndarrayPointer(self.data.reshape(4, -1), True)

    def lexico(self):
        return lexico(self.getHost(), [1, 2, 3, 4, 5])

    def ensurePureGauge(self):
        if self.pure_gauge is None:
            from .dirac.pure_gauge import PureGauge

            self.pure_gauge = PureGauge(self.latt_info)

    def gauss(self, seed: int, sigma: float):
        self.ensurePureGauge()
        self.pure_gauge.loadMom(self)
        self.pure_gauge.gaussMom(seed, sigma)
        self.pure_gauge.saveFreeMom(self)


class LatticeClover(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "Clover"))
        else:
            self.setData(value.reshape(2, Lt, Lz, Ly, Lx // 2, 2, ((Ns // 2) * Nc) ** 2))

    @property
    def data_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)


class LatticeFermion(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "Fermion"))
        else:
            self.setData(value.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc))

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
    def data_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)

    @property
    def even_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(2, -1)[0], True)

    @property
    def odd_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(2, -1)[1], True)

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])

    def timeslice(self, t: int):
        Lt = self.latt_info.Lt
        gt = self.latt_info.gt
        x = LatticeFermion(self.latt_info)
        if gt * Lt <= t < (gt + 1) * Lt:
            x.data[:, t - gt * Lt, :, :, :] = self.data[:, t - gt * Lt, :, :, :]
        return x


class MultiLatticeFermion(MultiLatticeField):
    def __init__(self, latt_info: LatticeInfo, L5: int, value=None) -> None:
        super().__init__(latt_info, L5)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newMultiLatticeFieldData(latt_info, L5, "Fermion"))
        else:
            self.setData(value.reshape(L5, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc))

    @property
    def data_ptrs(self) -> Pointers:
        return ndarrayPointer(self.data.reshape(self.L5, -1), True)

    @property
    def even_ptrs(self) -> Pointers:
        return ndarrayPointer(self.data.reshape(self.L5, 2, -1)[:, 0], True)

    @property
    def odd_ptrs(self) -> Pointers:
        return ndarrayPointer(self.data.reshape(self.L5, 2, -1)[:, 1], True)

    def __getitem__(self, index: int) -> LatticeFermion:
        return LatticeFermion(self.latt_info, self.data[index])

    def __setitem__(self, index: int, value: LatticeFermion):
        self.data[index] = value.data

    def lexico(self):
        return lexico(self.getHost(), [1, 2, 3, 4, 5])


class LatticePropagator(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "Propagator"))
        else:
            self.setData(value.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc))

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])

    def transpose(self):
        return self.data.transpose(0, 1, 2, 3, 4, 6, 5, 8, 7).copy()

    def setFermion(self, fermion: LatticeFermion, spin: int, color: int):
        self.data[:, :, :, :, :, :, spin, :, color] = fermion.data

    def getFermion(self, spin: int, color: int):
        return LatticeFermion(self.latt_info, self.data[:, :, :, :, :, :, spin, :, color])

    def timeslice(self, t: int):
        Lt = self.latt_info.Lt
        gt = self.latt_info.gt
        x = LatticePropagator(self.latt_info)
        if gt * Lt <= t < (gt + 1) * Lt:
            x.data[:, t - gt * Lt, :, :, :] = self.data[:, t - gt * Lt, :, :, :]
        return x


class LatticeStaggeredFermion(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "StaggeredFermion"))
        else:
            self.setData(value.reshape(2, Lt, Lz, Ly, Lx // 2, Nc))

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
    def data_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)

    @property
    def even_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(2, -1)[0], True)

    @property
    def odd_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(2, -1)[1], True)

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])

    def timeslice(self, t: int):
        Lt = self.latt_info.Lt
        gt = self.latt_info.gt
        x = LatticeStaggeredFermion(self.latt_info)
        if gt * Lt <= t < (gt + 1) * Lt:
            x.data[:, t - gt * Lt, :, :, :] = self.data[:, t - gt * Lt, :, :, :]
        return x


class MultiLatticeStaggeredFermion(MultiLatticeField):
    def __init__(self, latt_info: LatticeInfo, L5: int, value=None) -> None:
        super().__init__(latt_info, L5)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newMultiLatticeFieldData(latt_info, L5, "StaggeredFermion"))
        else:
            self.setData(value.reshape(L5, 2, Lt, Lz, Ly, Lx // 2, Nc))

    @property
    def data_ptrs(self) -> Pointers:
        return ndarrayPointer(self.data.reshape(self.L5, -1), True)

    @property
    def even_ptrs(self) -> Pointers:
        return ndarrayPointer(self.data.reshape(self.L5, 2, -1)[:, 0], True)

    @property
    def odd_ptrs(self) -> Pointers:
        return ndarrayPointer(self.data.reshape(self.L5, 2, -1)[:, 1], True)

    def __getitem__(self, index: int) -> LatticeStaggeredFermion:
        return LatticeStaggeredFermion(self.latt_info, self.data[index])

    def __setitem__(self, index: int, value: LatticeStaggeredFermion):
        self.data[index] = value.data

    def lexico(self):
        return lexico(self.getHost(), [1, 2, 3, 4, 5])


class LatticeStaggeredPropagator(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "StaggeredPropagator"))
        else:
            self.setData(value.reshape(2, Lt, Lz, Ly, Lx // 2, Nc, Nc))

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])

    def transpose(self):
        return self.data.transpose(0, 1, 2, 3, 4, 6, 5).copy()

    def setFermion(self, fermion: LatticeStaggeredFermion, color: int):
        self.data[:, :, :, :, :, :, color] = fermion.data

    def getFermion(self, color: int):
        return LatticeStaggeredFermion(self.latt_info, self.data[:, :, :, :, :, :, color])

    def timeslice(self, t: int):
        Lt = self.latt_info.Lt
        gt = self.latt_info.gt
        x = LatticeStaggeredPropagator(self.latt_info)
        if gt * Lt <= t < (gt + 1) * Lt:
            x.data[:, t - gt * Lt, :, :, :] = self.data[:, t - gt * Lt, :, :, :]
        return x
