from typing import Any, List, Literal, Sequence, Tuple, Union

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
        from . import init, isInitialized, getLogger, getGridSize

        if not isInitialized():
            init(None, latt_size)
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
        GLx, GLy, GLz, GLt = latt_size
        Lx, Ly, Lz, Lt = GLx // Gx, GLy // Gy, GLz // Gz, GLt // Gt

        self.Gx, self.Gy, self.Gz, self.Gt = Gx, Gy, Gz, Gt
        self.gx, self.gy, self.gz, self.gt = gx, gy, gz, gt
        self.GLx, self.GLy, self.GLz, self.GLt = GLx, GLy, GLz, GLt
        self.Lx, self.Ly, self.Lz, self.Lt = Lx, Ly, Lz, Lt
        self.global_size = [GLx, GLy, GLz, GLt]
        self.global_volume = GLx * GLy * GLz * GLt
        self.size = [Lx, Ly, Lz, Lt]
        self.volume = Lx * Ly * Lz * Lt
        self.ga_pad = Lx * Ly * Lz * Lt // min(Lx, Ly, Lz, Lt) // 2

        self.t_boundary = t_boundary
        self.anisotropy = anisotropy


class LaplaceLatticeInfo(LatticeInfo):
    def __init__(self, latt_size: List[int]):
        self._checkLatticeOddT(latt_size)
        self._setLattice(latt_size, 1, 1.0)

    def _checkLatticeOddT(self, latt_size: List[int]):
        from . import isInitialized, getLogger, getGridSize

        if not isInitialized():
            getLogger().critical("pyquda.init() must be called before contructing the LaplaceLatticeInfo", RuntimeError)
        Gx, Gy, Gz, Gt = getGridSize()
        Lx, Ly, Lz, Lt = latt_size
        if not (Lx % (2 * Gx) == 0 and Ly % (2 * Gy) == 0 and Lz % (2 * Gz) == 0 and Lt % Gt == 0):
            getLogger().critical(
                "lattice size must be divisible by gird size, "
                "and sublattice size must be even in spacial direction for consistant even-odd preconditioning",
                ValueError,
            )


Ns, Nc, Nd = LatticeInfo.Ns, LatticeInfo.Nc, LatticeInfo.Nd


class _Direction(int):
    def __new__(cls, x: int):
        return int.__new__(cls, x)

    def __neg__(self):
        return _Direction((self + 4) % 8)


X = _Direction(0)
Y = _Direction(1)
Z = _Direction(2)
T = _Direction(3)


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


class HalfLatticeField:
    def __init__(self, latt_info: LatticeInfo) -> None:
        from . import getCUDABackend

        self.latt_info = latt_info
        self._data = None
        self.backend: Literal["numpy", "cupy", "torch"] = getCUDABackend()
        self.L5 = None
        self.full_lattice = False

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
    def data_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)

    def setField(self, field_shape: Sequence[int], field_dtype: Literal["<c16", "<f8", "<i4"]):
        self.field_shape = tuple(field_shape)
        self.field_size = int(numpy.prod(field_shape))
        self.field_dtype = field_dtype
        Lx, Ly, Lz, Lt = self.latt_info.size
        self.lattice_shape = [2, Lt, Lz, Ly, Lx // 2] if self.full_lattice else [Lt, Lz, Ly, Lx // 2]
        self.shape = (
            (*self.lattice_shape, *self.field_shape)
            if self.L5 is None
            else (self.L5, *self.lattice_shape, *self.field_shape)
        )
        self.dtype = numpy.dtype(field_dtype).type
        if self.backend == "torch":
            from torch.testing._internal.common_utils import numpy_to_torch_dtype

            self.dtype = numpy_to_torch_dtype(self.dtype)

    def initData(self, value):
        backend, value = (value, None) if isinstance(value, str) else (None, value)
        if value is None:
            if backend == "numpy" or self.backend == "numpy":
                self.data = numpy.zeros(self.shape, self.dtype)
            elif self.backend == "cupy":
                import cupy

                self.data = cupy.zeros(self.shape, self.dtype)
            elif self.backend == "torch":
                import torch

                self.data = torch.zeros(self.shape, dtype=self.dtype)
        else:
            self.data = value.reshape(self.shape)

    @property
    def location(self) -> Literal["numpy", "cupy", "torch"]:
        if isinstance(self.data, numpy.ndarray):
            return "numpy"
        else:
            return self.backend

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

    def norm2(self, all_reduce=True) -> float:
        location = self.location
        if location == "numpy":
            norm2 = numpy.linalg.norm(self.data).item() ** 2
        elif location == "cupy":
            import cupy

            norm2 = cupy.linalg.norm(self.data).item() ** 2
        elif location == "torch":
            import torch

            norm2 = torch.linalg.norm(self.data).item() ** 2
        if all_reduce:
            return self.latt_info.mpi_comm.allreduce(norm2)
        else:
            return norm2

    def timeslice(self, start: int, stop: int = None, step: int = None, return_field: bool = True):
        Lt = self.latt_info.Lt
        gt = self.latt_info.gt
        stop = (start + 1) if stop is None else stop
        step = 1 if step is None else step
        if step > 0:
            s = (start - gt * Lt) % step if start < gt * Lt else 0
            start = max(start - gt * Lt, 0) + s
            stop = min(stop - gt * Lt, Lt)
        elif step < 0:
            s = ((gt + 1) * Lt - start) % step if (gt + 1) * Lt <= start else 0
            start = min(start - gt * Lt, Lt - 1) + s
            stop = max(stop - gt * Lt, -1)
            start, stop = (0, Lt) if start <= stop else (start, stop)
            stop = None if stop == -1 else stop  # Workaround for numpy slice
        if return_field:
            x = self.__class__(self.latt_info)
            if self.full_lattice and self.L5 is not None:
                x.data[:, :, start:stop:step, :, :, :] = self.data[:, :, start:stop:step, :, :, :]
            elif self.full_lattice or self.L5 is not None:
                x.data[:, start:stop:step, :, :, :] = self.data[:, start:stop:step, :, :, :]
            else:
                x.data[start:stop:step, :, :, :] = self.data[start:stop:step, :, :, :]
            return x
        else:
            if self.full_lattice and self.L5 is not None:
                return self.data[:, :, start:stop:step, :, :, :]
            elif self.full_lattice or self.L5 is not None:
                return self.data[:, start:stop:step, :, :, :]
            else:
                return self.data[start:stop:step, :, :, :]

    def __add__(self, other):
        assert self.__class__ == other.__class__ and self.location == other.location
        return self.__class__(self.latt_info, self.data + other.data)

    def __sub__(self, other):
        assert self.__class__ == other.__class__ and self.location == other.location
        return self.__class__(self.latt_info, self.data - other.data)

    def __mul__(self, other):
        return self.__class__(self.latt_info, self.data * other)

    def __rmul__(self, other):
        return self.__class__(self.latt_info, other * self.data)

    def __truediv__(self, other):
        return self.__class__(self.latt_info, self.data / other)

    def __neg__(self):
        return self.__class__(self.latt_info, -self.data)

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


class EvenOddField:
    def __init__(self, latt_info: LatticeInfo) -> None:
        s = super(EvenOddField, self)
        if hasattr(s, "__field_class__"):
            s.__field_class__.__base__.__init__(self, latt_info)
        else:
            s.__init__(latt_info)
        self.full_lattice = True

    @property
    def even(self):
        return super(EvenOddField, self).__field_class__(self.latt_info, self.data[0])

    @even.setter
    def even(self, value: HalfLatticeField):
        self.data[0] = value.data

    @property
    def odd(self):
        return super(EvenOddField, self).__field_class__(self.latt_info, self.data[1])

    @odd.setter
    def odd(self, value: HalfLatticeField):
        self.data[1] = value.data

    @property
    def even_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(2, -1)[0], True)

    @property
    def odd_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(2, -1)[1], True)

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])

    def _checksum(self, data) -> Tuple[int, int]:
        import zlib
        from mpi4py import MPI

        gx, gy, gz, gt = self.latt_info.grid_coord
        Lx, Ly, Lz, Lt = self.latt_info.size
        gLx, gLy, gLz, gLt = gx * Lx, gy * Ly, gz * Lz, gt * Lt
        GLx, GLy, GLz, GLt = self.latt_info.global_size
        work = numpy.empty((self.latt_info.volume), "<u4")
        for i in range(self.latt_info.volume):
            work[i] = zlib.crc32(data[i])
        rank = (
            numpy.arange(self.latt_info.global_volume, dtype="<u4")
            .reshape(GLt, GLz, GLy, GLx)[gLt : gLt + Lt, gLz : gLz + Lz, gLy : gLy + Ly, gLx : gLx + Lx]
            .reshape(-1)
        )
        rank29 = rank % 29
        rank31 = rank % 31
        sum29 = self.latt_info.mpi_comm.allreduce(
            numpy.bitwise_xor.reduce(work << rank29 | work >> (32 - rank29)).item(), MPI.BXOR
        )
        sum31 = self.latt_info.mpi_comm.allreduce(
            numpy.bitwise_xor.reduce(work << rank31 | work >> (32 - rank31)).item(), MPI.BXOR
        )
        return sum29, sum31

    def checksum(self, big_endian: bool = False) -> Tuple[int, int]:
        data = self.lexico() if not big_endian else self.lexico().astype(f">{self.field_dtype[1:]}")
        return self._checksum(data)


class MultiField:
    def __init__(self, latt_info: LatticeInfo, L5: int) -> None:
        s = super(MultiField, self)
        if hasattr(s, "__field_class__"):
            s.__field_class__.__base__.__init__(self, latt_info)
        else:
            s.__init__(latt_info)
        self.L5 = L5

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        contiguous = True
        for index in range(self.L5):
            if isinstance(value, numpy.ndarray) or self.backend == "numpy":
                contiguous &= value[index].flags.c_contiguous
            elif self.backend == "cupy":
                contiguous &= value[index].flags.c_contiguous
            elif self.backend == "torch":
                contiguous &= value[index].is_contiguous()
        if contiguous:
            self._data = value
        else:
            if isinstance(value, numpy.ndarray) or self.backend == "numpy":
                self._data = numpy.ascontiguousarray(value)
            elif self.backend == "cupy":
                import cupy

                self._data = cupy.ascontiguousarray(value)
            elif self.backend == "torch":
                self._data = value.contiguous()

    def __getitem__(self, key: Union[int, list, tuple, slice]):
        if isinstance(key, int):
            return super(MultiField, self).__field_class__(self.latt_info, self.data[key])
        elif isinstance(key, list):
            return self.__class__(self.latt_info, len(key), self.data[key])
        elif isinstance(key, tuple):
            return self.__class__(self.latt_info, len(key), self.data[list(key)])
        elif isinstance(key, slice):
            return self.__class__(self.latt_info, len(range(*key.indices(self.L5))), self.data[key])

    def __setitem__(self, key: Union[int, list, tuple, slice], value):
        self.data[key] = value.data

    def even(self, index: int):
        return super(EvenOddField, self).__field_class__(self.latt_info, self.data[index, 0])

    def odd(self, index: int):
        return super(EvenOddField, self).__field_class__(self.latt_info, self.data[index, 1])

    def data_ptr(self, index: int = 0) -> Pointer:
        return ndarrayPointer(self.data.reshape(self.L5, -1)[index], True)

    def even_ptr(self, index: int) -> Pointer:
        assert self.full_lattice
        return ndarrayPointer(self.data.reshape(self.L5, 2, -1)[index, 0], True)

    def odd_ptr(self, index: int) -> Pointer:
        assert self.full_lattice
        return ndarrayPointer(self.data.reshape(self.L5, 2, -1)[index, 1], True)

    @property
    def data_ptrs(self) -> Pointers:
        return ndarrayPointer(self.data.reshape(self.L5, -1), True)

    @property
    def even_ptrs(self) -> Pointers:
        assert self.full_lattice
        return ndarrayPointer(self.data.reshape(self.L5, 2, -1)[:, 0], True)

    @property
    def odd_ptrs(self) -> Pointers:
        assert self.full_lattice
        return ndarrayPointer(self.data.reshape(self.L5, 2, -1)[:, 1], True)

    def copy(self):
        return self.__class__(self.latt_info, self.L5, self.backup())

    def lexico(self):
        return lexico(self.getHost(), [1, 2, 3, 4, 5])

    def checksum(self, big_endian: bool = False) -> Tuple[int, int]:
        assert self.full_lattice
        data = (
            (
                self.lexico()
                .reshape(self.L5, self.latt_info.volume, self.field_size)
                .transpose(1, 0, 2)
                .reshape(self.latt_info.volume, self.L5 * self.field_size)
                .copy()
            )
            if not big_endian
            else (
                self.lexico()
                .reshape(self.L5, self.latt_info.volume, self.field_size)
                .transpose(1, 0, 2)
                .reshape(self.latt_info.volume, self.L5 * self.field_size)
                .astype(f">{self.field_dtype[1:]}")
            )
        )
        return self._checksum(data)

    def __add__(self, other):
        assert self.__class__ == other.__class__ and self.location == other.location
        return self.__class__(self.latt_info, self.L5, self.data + other.data)

    def __sub__(self, other):
        assert self.__class__ == other.__class__ and self.location == other.location
        return self.__class__(self.latt_info, self.L5, self.data - other.data)

    def __mul__(self, other):
        return self.__class__(self.latt_info, self.L5, self.data * other)

    def __rmul__(self, other):
        return self.__class__(self.latt_info, self.L5, other * self.data)

    def __truediv__(self, other):
        return self.__class__(self.latt_info, self.L5, self.data / other)

    def __neg__(self):
        return self.__class__(self.latt_info, self.L5, -self.data)


class LatticeInt32(EvenOddField, HalfLatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([], "<i4")
        self.initData(value)


class LatticeFloat64(EvenOddField, HalfLatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([], "<f8")
        self.initData(value)


class LatticeComplex128(EvenOddField, HalfLatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([], "<c16")
        self.initData(value)


class LatticeLink(EvenOddField, HalfLatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([Nc, Nc], "<c16")
        self.initData(value)
        if value is None:
            if self.backend == "numpy":
                self.data[:] = numpy.identity(Nc)
            elif self.backend == "cupy":
                import cupy

                self.data[:] = cupy.identity(Nc)
            elif self.backend == "torch":
                import torch

                self.data[:] = torch.eye(Nc)

    @property
    def __field_class__(self):
        return LatticeLink

    def pack(self, x: "LatticeFermion"):
        for color in range(Nc):
            x.data[:, :, :, :, :, color, :] = self.data[:, :, :, :, :, :, color]

    def unpack(self, x: "LatticeFermion"):
        for color in range(Nc):
            self.data[:, :, :, :, :, :, color] = x.data[:, :, :, :, :, color, :]


class LatticeGauge(MultiField, LatticeLink):
    def __init__(self, latt_info: LatticeInfo, L5: Union[int, Any] = Nd, value=None) -> None:
        """`L5` can be `value` here"""
        if not isinstance(L5, int):
            value = L5
            L5 = Nd
        super().__init__(latt_info, L5)
        self.setField([Nc, Nc], "<c16")
        self.initData(value)
        if value is None:
            if self.backend == "numpy":
                self.data[:] = numpy.identity(Nc)
            elif self.backend == "cupy":
                import cupy

                self.data[:] = cupy.identity(Nc)
            elif self.backend == "torch":
                import torch

                self.data[:] = torch.eye(Nc)
        self._gauge_dirac = None

    @property
    def gauge_dirac(self):
        if self._gauge_dirac is None:
            from .dirac import GaugeDirac

            self._gauge_dirac = GaugeDirac(self.latt_info)
        return self._gauge_dirac

    @property
    def pure_gauge(self):
        return self.gauge_dirac

    def setAntiPeriodicT(self):
        if self.latt_info.gt == self.latt_info.Gt - 1:
            self.data[Nd - 1, :, self.latt_info.Lt - 1] *= -1

    def setAnisotropy(self, anisotropy: float):
        self.data[: Nd - 1] /= anisotropy

    def ensurePureGauge(self):
        pass

    def covDev(self, x: "LatticeFermion", covdev_mu: int):
        self.gauge_dirac.loadGauge(self)
        b = self._gauge_dirac.covDev(x, covdev_mu)
        self._gauge_dirac.freeGauge()
        return b

    def laplace(self, x: "LatticeStaggeredFermion", laplace3D: int):
        self.gauge_dirac.loadGauge(self)
        b = self._gauge_dirac.laplace(x, laplace3D)
        self._gauge_dirac.freeGauge()
        return b

    def wuppertalSmear(self, x: Union["LatticeFermion", "LatticeStaggeredFermion"], n_steps: int, alpha: float):
        self.gauge_dirac.loadGauge(self)
        b = self._gauge_dirac.wuppertalSmear(x, n_steps, alpha)
        self._gauge_dirac.freeGauge()
        return b

    def shift(self, shift_mu: List[int]):
        unit = LatticeGauge(self.latt_info)
        x = LatticeFermion(self.latt_info)
        self.gauge_dirac.loadGauge(unit)
        for mu, covdev_mu in enumerate(shift_mu):
            self[mu].pack(x)
            b = self._gauge_dirac.covDev(x, covdev_mu)
            unit[mu].unpack(b)
            # x.data[:, :, :, :, :, :Nc, :Nc] = self.data[mu]
            # unit.data[mu] = self._gauge_dirac.covDev(x, covdev_mu).data[:, :, :, :, :, :Nc, :Nc]
        self._gauge_dirac.freeGauge()
        return unit

    def staggeredPhase(self, applied: bool):
        self.gauge_dirac.staggeredPhase(self, applied)

    def projectSU3(self, tol: float):
        self.gauge_dirac.projectSU3(self, tol)

    def path(self, paths: List[List[int]]):
        return self.gauge_dirac.path(self, paths)

    def loop(self, loops: List[List[List[int]]], coeff: List[float]):
        return self.gauge_dirac.loop(self, loops, coeff)

    def loopTrace(self, loops: List[List[int]]):
        self.gauge_dirac.loadGauge(self)
        traces = self._gauge_dirac.loopTrace(loops)
        self._gauge_dirac.freeGauge()
        return traces

    def apeSmear(self, n_steps: int, alpha: float, dir_ignore: int):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.apeSmear(n_steps, alpha, dir_ignore)
        self._gauge_dirac.saveSmearedGauge(self)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()

    def stoutSmear(self, n_steps: int, rho: float, dir_ignore: int):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.stoutSmear(n_steps, rho, dir_ignore)
        self._gauge_dirac.saveSmearedGauge(self)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()

    def hypSmear(self, n_steps: int, alpha1: float, alpha2: float, alpha3: float, dir_ignore: int):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore)
        self._gauge_dirac.saveSmearedGauge(self)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()

    def wilsonFlow(self, n_steps: int, epsilon: float):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.wilsonFlow(1, epsilon, 0, False)
        energy = [self._gauge_dirac.obs_param.energy]
        for step in range(1, n_steps):
            self._gauge_dirac.wilsonFlow(1, epsilon, step * epsilon, True)
            energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.saveSmearedGauge(self)  # Save before the last step
        self._gauge_dirac.wilsonFlow(1, epsilon, n_steps * epsilon, True)
        energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return energy

    def wilsonFlowScale(self, max_steps: int, epsilon: float):
        from . import getLogger

        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.wilsonFlow(1, epsilon, 0, False)
        t2E, tdt2E = 0, 0
        t0, w0 = 0, 0
        for step in range(1, max_steps + 1):
            if t2E >= 0.3 and tdt2E >= 0.3:
                break
            self._gauge_dirac.wilsonFlow(1, epsilon, step * epsilon, True)
            t2E_old, t2E = t2E, (step * epsilon) ** 2 * self._gauge_dirac.obs_param.energy[0]
            tdt2E_old, tdt2E = tdt2E, (step - 0.5) * (t2E - t2E_old)
            if t0 == 0 and t2E >= 0.3:
                t0 = (step - (t2E - 0.3) / (t2E - t2E_old)) * epsilon
            if w0 == 0 and tdt2E >= 0.3:
                w0 = ((step - 0.5 - (tdt2E - 0.3) / (tdt2E - tdt2E_old)) * epsilon) ** 0.5
            getLogger().info(f"t2E({step * epsilon})={t2E}, tdt2E({(step - 0.5) * epsilon})={tdt2E}")
        else:
            getLogger().error(
                f"Wilson flow scale doesn't exceed 0.3 at max_steps*epsilon={max_steps * epsilon}", RuntimeError
            )
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return t0, w0

    def symanzikFlow(self, n_steps: int, epsilon: float):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.symanzikFlow(1, epsilon, 0, False)
        energy = [self._gauge_dirac.obs_param.energy]
        for step in range(1, n_steps):
            self._gauge_dirac.symanzikFlow(1, epsilon, step * epsilon, True)
            energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.saveSmearedGauge(self)  # Save before the last step
        self._gauge_dirac.symanzikFlow(1, epsilon, n_steps * epsilon, True)
        energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return energy

    def symanzikFlowScale(self, max_steps: int, epsilon: float):
        from . import getLogger

        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.symanzikFlow(1, epsilon, 0, False)
        t2E, tdt2E = 0, 0
        t0, w0 = 0, 0
        for step in range(1, max_steps + 1):
            if t2E >= 0.3 and tdt2E >= 0.3:
                break
            self._gauge_dirac.symanzikFlow(1, epsilon, step * epsilon, True)
            t2E_old, t2E = t2E, (step * epsilon) ** 2 * self._gauge_dirac.obs_param.energy[0]
            tdt2E_old, tdt2E = tdt2E, (step - 0.5) * (t2E - t2E_old)
            if t0 == 0 and t2E >= 0.3:
                t0 = (step - (t2E - 0.3) / (t2E - t2E_old)) * epsilon
            if w0 == 0 and tdt2E >= 0.3:
                w0 = ((step - 0.5 - (tdt2E - 0.3) / (tdt2E - tdt2E_old)) * epsilon) ** 0.5
            getLogger().info(f"t2E({step * epsilon})={t2E}, tdt2E({(step - 0.5) * epsilon})={tdt2E}")
        else:
            getLogger().error(
                f"Symanzik flow scale doesn't exceed 0.3 at max_steps*epsilon={max_steps * epsilon}", RuntimeError
            )
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
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
        self.gauge_dirac.loadGauge(self)
        plaquette = self._gauge_dirac.plaquette()
        self._gauge_dirac.freeGauge()
        return plaquette

    def polyakovLoop(self):
        self.gauge_dirac.loadGauge(self)
        polyakovLoop = self._gauge_dirac.polyakovLoop()
        self._gauge_dirac.freeGauge()
        return polyakovLoop

    def energy(self):
        self.gauge_dirac.loadGauge(self)
        energy = self._gauge_dirac.energy()
        self._gauge_dirac.freeGauge()
        return energy

    def qcharge(self):
        self.gauge_dirac.loadGauge(self)
        qcharge = self._gauge_dirac.qcharge()
        self._gauge_dirac.freeGauge()
        return qcharge

    def qchargeDensity(self):
        self.gauge_dirac.loadGauge(self)
        qcharge_density = self._gauge_dirac.qchargeDensity()
        self._gauge_dirac.freeGauge()
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
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.gaussGauge(seed, sigma)
        self._gauge_dirac.saveGauge(self)
        self._gauge_dirac.freeGauge()

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
        self.gauge_dirac.fixingOVR(
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
        self.gauge_dirac.fixingFFT(self, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta)


class LatticeMom(MultiField, EvenOddField, HalfLatticeField):
    def __init__(self, latt_info: LatticeInfo, L5: Union[int, Any] = Nd, value=None) -> None:
        """`L5` can be `value` here"""
        if not isinstance(L5, int):
            value = L5
            L5 = Nd
        super().__init__(latt_info, L5)
        self.setField([10], "<f8")
        self.initData(value)
        self._gauge_dirac = None

    @property
    def gauge_dirac(self):
        if self._gauge_dirac is None:
            from .dirac import GaugeDirac

            self._gauge_dirac = GaugeDirac(self.latt_info)
        return self._gauge_dirac

    def lexico(self):
        return lexico(self.getHost(), [1, 2, 3, 4, 5])

    def gauss(self, seed: int, sigma: float):
        self.gauge_dirac.loadMom(self)
        self._gauge_dirac.gaussMom(seed, sigma)
        self._gauge_dirac.saveFreeMom(self)


class LatticeClover(EvenOddField, HalfLatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([2, ((Ns // 2) * Nc) ** 2], "<f8")
        self.initData(value)


class HalfLatticeFermion(HalfLatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([Ns, Nc], "<c16")
        self.initData(value)

    @property
    def __field_class__(self):
        return HalfLatticeFermion


class LatticeFermion(EvenOddField, HalfLatticeFermion):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([Ns, Nc], "<c16")
        self.initData(value)

    @property
    def __field_class__(self):
        return LatticeFermion


class MultiHalfLatticeFermion(MultiField, HalfLatticeFermion):
    def __init__(self, latt_info: LatticeInfo, L5: int, value=None) -> None:
        super().__init__(latt_info, L5)
        self.setField([Ns, Nc], "<c16")
        self.initData(value)


class MultiLatticeFermion(MultiField, LatticeFermion):
    def __init__(self, latt_info: LatticeInfo, L5: int, value=None) -> None:
        super().__init__(latt_info, L5)
        self.setField([Ns, Nc], "<c16")
        self.initData(value)

    def toPropagator(self):
        assert self.L5 == Ns * Nc
        return LatticePropagator(
            self.latt_info,
            self.data.reshape(Ns, Nc, *self.lattice_shape, Ns, Nc).transpose(2, 3, 4, 5, 6, 7, 0, 8, 1),
        )


class LatticePropagator(EvenOddField, HalfLatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([Ns, Ns, Nc, Nc], "<c16")
        self.initData(value)

    def setFermion(self, fermion: LatticeFermion, spin: int, color: int):
        self.data[:, :, :, :, :, :, spin, :, color] = fermion.data

    def getFermion(self, spin: int, color: int):
        return LatticeFermion(self.latt_info, self.data[:, :, :, :, :, :, spin, :, color])

    def toMultiFermion(self):
        return MultiLatticeFermion(
            self.latt_info,
            Ns * Nc,
            self.data.transpose(6, 8, 0, 1, 2, 3, 4, 5, 7).reshape(Ns * Nc, *self.lattice_shape, Ns, Nc),
        )


class HalfLatticeStaggeredFermion(HalfLatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([Nc], "<c16")
        self.initData(value)

    @property
    def __field_class__(self):
        return HalfLatticeStaggeredFermion


class MultiHalfLatticeStaggeredFermion(MultiField, HalfLatticeStaggeredFermion):
    def __init__(self, latt_info: LatticeInfo, L5: int, value=None) -> None:
        super().__init__(latt_info, L5)
        self.setField([Nc], "<c16")
        self.initData(value)


class LatticeStaggeredFermion(EvenOddField, HalfLatticeStaggeredFermion):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([Nc], "<c16")
        self.initData(value)

    @property
    def __field_class__(self):
        return LatticeStaggeredFermion


class MultiLatticeStaggeredFermion(MultiField, LatticeStaggeredFermion):
    def __init__(self, latt_info: LatticeInfo, L5: int, value=None) -> None:
        super().__init__(latt_info, L5)
        self.setField([Nc], "<c16")
        self.initData(value)

    def toPropagator(self):
        assert self.L5 == Nc
        return LatticeStaggeredPropagator(
            self.latt_info,
            self.data.reshape(Nc, *self.lattice_shape, Nc).transpose(1, 2, 3, 4, 5, 6, 0),
        )


class LatticeStaggeredPropagator(EvenOddField, HalfLatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        self.setField([Nc, Nc], "<c16")
        self.initData(value)

    def setFermion(self, fermion: LatticeStaggeredFermion, color: int):
        self.data[:, :, :, :, :, :, color] = fermion.data

    def getFermion(self, color: int) -> LatticeStaggeredFermion:
        return LatticeStaggeredFermion(self.latt_info, self.data[:, :, :, :, :, :, color])

    def toMultiFermion(self):
        return MultiLatticeStaggeredFermion(
            self.latt_info,
            Nc,
            self.data.transpose(6, 0, 1, 2, 3, 4, 5).reshape(Nc, *self.lattice_shape, Nc),
        )
