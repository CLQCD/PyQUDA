from abc import abstractmethod
from os import path
from time import perf_counter
from typing import Any, List, Literal, Sequence, Tuple, Union

import numpy
from numpy.typing import NDArray

from . import (
    getLogger,
    getMPIComm,
    getMPISize,
    getMPIRank,
    getGridSize,
    getGridCoord,
    getCUDABackend,
)


class LatticeInfo:
    Nd: int = 4
    Ns: int = 4
    Nc: int = 3

    def __init__(self, latt_size: List[int], t_boundary: Literal[1, -1] = 1, anisotropy: float = 1.0) -> None:
        self._checkLattice(latt_size)
        self._setLattice(latt_size, t_boundary, anisotropy)

    def _checkLattice(self, latt_size: List[int]):
        GLx, GLy, GLz, GLt = latt_size
        Gx, Gy, Gz, Gt = getGridSize()
        if not (
            (GLx % (2 * Gx) == 0 or GLx * Gx == 1)
            and (GLy % (2 * Gy) == 0 or GLy * Gy == 1)
            and (GLz % (2 * Gz) == 0 or GLz * Gz == 1)
            and (GLt % (2 * Gt) == 0 or GLt * Gt == 1)
        ):
            getLogger().critical(
                "lattice size must be divisible by gird size, "
                "and sublattice size must be even in all directions for consistant even-odd preconditioning, ",
                "otherwise the lattice size and grid size for this direction must be 1",
                ValueError,
            )

    def _setLattice(self, latt_size: List[int], t_boundary: Literal[1, -1], anisotropy: float):
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


Nd, Ns, Nc = LatticeInfo.Nd, LatticeInfo.Ns, LatticeInfo.Nc


class GeneralInfo:
    def __init__(self, latt_size: List[int], Ns: int = 4, Nc: int = 3) -> None:
        grid_size = getGridSize()
        self._checkLattice(latt_size, grid_size)
        self._setLattice(latt_size, grid_size)
        self.Nd = len(latt_size)
        self.Ns = Ns
        self.Nc = Nc

    def _checkLattice(self, latt_size: List[int], grid_size: List[int]):
        assert len(latt_size) == len(grid_size), "lattice size and grid size must have the same dimension"
        for GL, G in zip(latt_size, grid_size):
            if not (GL % G == 0):
                getLogger().critical("lattice size must be divisible by gird size", ValueError)

    def _setLattice(self, latt_size: List[int], grid_size: List[int]):
        self.mpi_comm = getMPIComm()
        self.mpi_size = getMPISize()
        self.mpi_rank = getMPIRank()
        self.grid_size = grid_size
        self.grid_coord = getGridCoord()

        self.global_size = latt_size
        self.global_volume = int(numpy.prod(latt_size))
        self.size = [GL // G for GL, G in zip(latt_size, grid_size)]
        self.volume = int(numpy.prod(self.size))


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
    data_evenodd = data.reshape(Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf)
    data_lexico = numpy.zeros((Npre, Lt, Lz, Ly, Lx, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_lexico[:, t, z, y, 0::2] = data_evenodd[:, 0, t, z, y, :]
                    data_lexico[:, t, z, y, 1::2] = data_evenodd[:, 1, t, z, y, :]
                else:
                    data_lexico[:, t, z, y, 1::2] = data_evenodd[:, 0, t, z, y, :]
                    data_lexico[:, t, z, y, 0::2] = data_evenodd[:, 1, t, z, y, :]
    return data_lexico.reshape(*shape[: axes[0]], Lt, Lz, Ly, Lx, *shape[axes[-1] + 1 :])


def evenodd(data: numpy.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    Npre = int(numpy.prod(shape[: axes[0]]))
    Nsuf = int(numpy.prod(shape[axes[-1] + 1 :]))
    dtype = data.dtype if dtype is None else dtype
    data_lexico = data.reshape(Npre, Lt, Lz, Ly, Lx, Nsuf)
    data_evenodd = numpy.zeros((Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_evenodd[:, 0, t, z, y, :] = data_lexico[:, t, z, y, 0::2]
                    data_evenodd[:, 1, t, z, y, :] = data_lexico[:, t, z, y, 1::2]
                else:
                    data_evenodd[:, 0, t, z, y, :] = data_lexico[:, t, z, y, 1::2]
                    data_evenodd[:, 1, t, z, y, :] = data_lexico[:, t, z, y, 0::2]
    return data_evenodd.reshape(*shape[: axes[0]], 2, Lt, Lz, Ly, Lx // 2, *shape[axes[-1] + 1 :])


def cb2(data: numpy.ndarray, axes: List[int], dtype=None):
    getLogger().warning("cb2 is deprecated, use evenodd instead", DeprecationWarning)
    return evenodd(data, axes, dtype)


def checksum(latt_info: Union[LatticeInfo, GeneralInfo], data: numpy.ndarray) -> Tuple[int, int]:
    import zlib
    from mpi4py import MPI

    work = numpy.empty((latt_info.volume), "<u4")
    for i in range(latt_info.volume):
        work[i] = zlib.crc32(data[i])
    sublatt_slice = tuple(slice(g * L, (g + 1) * L) for g, L in zip(latt_info.grid_coord[::-1], latt_info.size[::-1]))
    rank = (
        numpy.arange(latt_info.global_volume, dtype="<u8")
        .reshape(*latt_info.global_size[::-1])[sublatt_slice]
        .reshape(-1)
    )
    rank29 = (rank % 29).astype("<u4")
    rank31 = (rank % 31).astype("<u4")
    sum29 = latt_info.mpi_comm.allreduce(
        numpy.bitwise_xor.reduce(work << rank29 | work >> (32 - rank29)).item(), MPI.BXOR
    )
    sum31 = latt_info.mpi_comm.allreduce(
        numpy.bitwise_xor.reduce(work << rank31 | work >> (32 - rank31)).item(), MPI.BXOR
    )
    return sum29, sum31


def _field_shape_dtype(field: str, Ns: int, Nc: int, use_fp32: bool = False):
    float_nbytes = 4 if use_fp32 else 8
    if field in ["Int"]:
        return [], "<i4"
    elif field in ["Real"]:
        return [], f"<f{float_nbytes}"
    elif field in ["Complex"]:
        return [], f"<c{2 * float_nbytes}"
    elif field in ["SpinColorVector"]:
        return [Ns, Nc], f"<c{2 * float_nbytes}"
    elif field in ["SpinColorMatrix"]:
        return [Ns, Ns, Nc, Nc], f"<c{2 * float_nbytes}"
    elif field in ["ColorVector"]:
        return [Nc], f"<c{2 * float_nbytes}"
    elif field in ["ColorMatrix"]:
        return [Nc, Nc], f"<c{2 * float_nbytes}"
    elif field in ["Mom"]:
        return [Nc**2 + 1], f"<f{float_nbytes}"
    elif field in ["Clover"]:
        return [2, ((Ns // 2) * Nc) ** 2], f"<f{float_nbytes}"
    else:
        getLogger().critical(f"Unknown field type: {field}", ValueError)


class BaseField:
    def __init__(self, latt_info: Union[LatticeInfo, GeneralInfo]) -> None:
        self.latt_info = latt_info
        self._data = None
        self.backend: Literal["numpy", "cupy", "torch"] = getCUDABackend()
        self.L5 = None

    @abstractmethod
    def _shape(self):
        getLogger().critical("_setShape method must be implemented", NotImplementedError)

    @classmethod
    def _groupName(cls):
        if cls.__name__ == "LatticeMom":
            getLogger().critical("LatticeMom is not supported for save/load", ValueError)
        elif cls.__name__ == "LatticeClover":
            getLogger().critical("LatticeClover is not supported for save/load", ValueError)

        return (
            cls.__name__.replace("Multi", "")
            .replace("General", "")
            .replace("Link", "ColorMatrix")
            .replace("Gauge", "ColorMatrix")
            .replace("StaggeredFermion", "ColorVector")
            .replace("StaggeredPropagator", "ColorMatrix")
            .replace("Fermion", "SpinColorVector")
            .replace("Propagator", "SpinColorMatrix")
        )

    def save(
        self,
        filename: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        *,
        annotation: str = "",
        check: bool = True,
        use_fp32: bool = False,
    ):
        from .file import File

        assert hasattr(self, "lexico")
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        with File(filename, "w") as f:
            f.save(
                self._groupName(),
                label,
                self.lexico(),
                annotation=annotation,
                check=check,
                use_fp32=use_fp32,
            )
        secs = perf_counter() - s
        getLogger().debug(f"Saved {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")

    def append(
        self,
        filename: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        *,
        annotation: str = "",
        check: bool = True,
        use_fp32: bool = False,
    ):
        from .file import File

        assert hasattr(self, "lexico")
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        with File(filename, "r+") as f:
            f.append(
                self._groupName(),
                label,
                self.lexico(),
                annotation=annotation,
                check=check,
                use_fp32=use_fp32,
            )
        secs = perf_counter() - s
        getLogger().debug(f"Appended {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")

    def update(
        self,
        filename: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        *,
        annotation: str = "",
        check: bool = True,
    ):
        from .file import File

        assert hasattr(self, "lexico")
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        with File(filename, "r+") as f:
            f.update(
                self._groupName(),
                label,
                self.lexico(),
                annotation=annotation,
                check=check,
            )
        secs = perf_counter() - s
        getLogger().debug(f"Updated {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")

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
    def data_ptr(self) -> NDArray:
        return self.data.reshape(-1)

    @classmethod
    def _field(cls) -> str:
        group_name = cls._groupName()
        return group_name[group_name.index("Lattice") + len("Lattice") :]

    def _setField(self):
        field_shape, field_dtype = _field_shape_dtype(self._field(), self.latt_info.Ns, self.latt_info.Nc)
        self.field_shape = tuple(field_shape)
        self.field_size = int(numpy.prod(field_shape))
        self.field_dtype = field_dtype
        self.shape = self._shape()
        self.dtype = numpy.dtype(field_dtype).type
        if self.backend == "torch":
            import torch

            # from torch.testing._internal.common_utils import numpy_to_torch_dtype_dict
            # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
            numpy_to_torch_dtype_dict = {
                numpy.bool_: torch.bool,
                numpy.uint8: torch.uint8,
                numpy.uint16: torch.uint16,
                numpy.uint32: torch.uint32,
                numpy.uint64: torch.uint64,
                numpy.int8: torch.int8,
                numpy.int16: torch.int16,
                numpy.int32: torch.int32,
                numpy.int64: torch.int64,
                numpy.float16: torch.float16,
                numpy.float32: torch.float32,
                numpy.float64: torch.float64,
                numpy.complex64: torch.complex64,
                numpy.complex128: torch.complex128,
            }
            self.dtype = numpy_to_torch_dtype_dict[self.dtype]

    def _initData(self, value):
        self._setField()
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


class GeneralField(BaseField):
    def __init__(self, latt_info: GeneralInfo, value: Any = None, init_data: bool = True) -> None:
        super().__init__(latt_info)
        if init_data:
            self._initData(value)

    def _shape(self):
        self.lattice_shape = self.latt_info.size[::-1]
        if self.L5 is None:
            return (*self.lattice_shape, *self.field_shape)
        else:
            return (self.L5, *self.lattice_shape, *self.field_shape)

    def lexico(self, dtype=None):
        return self.getHost().astype(dtype)

    def checksum(self) -> Tuple[int, int]:
        return checksum(self.latt_info, self.lexico().reshape(self.latt_info.volume, self.field_size).view("<u4"))

    @classmethod
    def load(
        cls,
        filename: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        *,
        check: bool = True,
    ):
        from .file import File

        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        with File(filename, "r") as f:
            latt_size, Ns, Nc, value = f.load(cls._groupName(), label, check=check)
        latt_info = GeneralInfo(latt_size, Ns, Nc)
        if not issubclass(cls, MultiField):
            retval = cls(latt_info, value)
        else:
            retval = cls(latt_info, len(label), numpy.asarray(value))
        secs = perf_counter() - s
        getLogger().debug(f"Loaded {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")
        return retval


class ParityField(BaseField):
    def __init__(self, latt_info: LatticeInfo, value: Any = None, init_data: bool = True) -> None:
        super().__init__(latt_info)
        self.full_field = False
        if init_data:
            self._initData(value)

    def _shape(self):
        Lx, Ly, Lz, Lt = self.latt_info.size
        self.lattice_shape = [2, Lt, Lz, Ly, Lx // 2] if self.full_field else [Lt, Lz, Ly, Lx // 2]
        if self.L5 is None:
            return (*self.lattice_shape, *self.field_shape)
        else:
            return (self.L5, *self.lattice_shape, *self.field_shape)

    def timeslice(self, start: int, stop: int = None, step: int = None, return_field: bool = True):
        Lt = self.latt_info.Lt
        gt = self.latt_info.gt
        stop = (start + 1) if stop is None else stop
        step = 1 if step is None else step
        s = (start - gt * Lt) % step if start < gt * Lt and stop > gt * Lt else 0
        start = min(max(start - gt * Lt, 0), Lt) + s
        stop = min(max(stop - gt * Lt, 0), Lt)
        assert start <= stop and step > 0
        if return_field:
            if self.L5 is None:
                x = self.__class__(self.latt_info)
            else:
                x = self.__class__(self.latt_info, self.L5)
            if self.full_field and self.L5 is not None:
                x.data[:, :, start:stop:step, :, :, :] = self.data[:, :, start:stop:step, :, :, :]
            elif self.full_field or self.L5 is not None:
                x.data[:, start:stop:step, :, :, :] = self.data[:, start:stop:step, :, :, :]
            else:
                x.data[start:stop:step, :, :, :] = self.data[start:stop:step, :, :, :]
        else:
            if self.full_field and self.L5 is not None:
                x = self.data[:, :, start:stop:step, :, :, :]
            elif self.full_field or self.L5 is not None:
                x = self.data[:, start:stop:step, :, :, :]
            else:
                x = self.data[start:stop:step, :, :, :]
        return x


class FullField:
    latt_info: LatticeInfo

    def __init__(self, latt_info: LatticeInfo, value: Any = None, init_data: bool = True) -> None:
        s = super(FullField, self)
        if hasattr(s, "__field_class__"):
            s.__field_class__.__base__.__init__(self, latt_info, value, False)
        else:
            s.__init__(latt_info, value, False)
        self.full_field = True
        if init_data:
            self._initData(value)

    @property
    def even(self):
        return super(FullField, self).__field_class__(self.latt_info, self.data[0])

    @even.setter
    def even(self, value: ParityField):
        self.data[0] = value.data

    @property
    def odd(self):
        return super(FullField, self).__field_class__(self.latt_info, self.data[1])

    @odd.setter
    def odd(self, value: ParityField):
        self.data[1] = value.data

    @property
    def even_ptr(self) -> NDArray:
        return self.data.reshape(2, -1)[0]

    @property
    def odd_ptr(self) -> NDArray:
        return self.data.reshape(2, -1)[1]

    def lexico(self, dtype=None):
        return lexico(self.getHost(), [0, 1, 2, 3, 4], dtype)

    def checksum(self) -> Tuple[int, int]:
        return checksum(self.latt_info, self.lexico().reshape(self.latt_info.volume, self.field_size).view("<u4"))

    @classmethod
    def load(
        cls,
        filename: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        *,
        check: bool = True,
    ):
        from .file import File

        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        with File(filename, "r") as f:
            latt_size, Ns, Nc, value = f.load(cls._groupName(), label, check=check)
        latt_info = LatticeInfo(latt_size)
        if Ns is not None:
            latt_info.Ns = Ns
        if Nc is not None:
            latt_info.Nc = Nc
        if not issubclass(cls, MultiField):
            retval = cls(latt_info, evenodd(value, [0, 1, 2, 3]))
        else:
            retval = cls(latt_info, len(label), numpy.asarray([evenodd(data, [0, 1, 2, 3]) for data in value]))
        secs = perf_counter() - s
        getLogger().debug(f"Loaded {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")
        return retval


class MultiField:
    latt_info: LatticeInfo

    def __init__(self, latt_info: LatticeInfo, L5: int, value: Any = None, init_data: bool = True) -> None:
        assert L5 > 0
        s = super(MultiField, self)
        if hasattr(s, "__field_class__"):
            s.__field_class__.__base__.__init__(self, latt_info, value, False)
        else:
            s.__init__(latt_info, value, False)
        self.L5 = L5
        if init_data:
            self._initData(value)

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
        return super(FullField, self).__field_class__(self.latt_info, self.data[index, 0])

    def odd(self, index: int):
        return super(FullField, self).__field_class__(self.latt_info, self.data[index, 1])

    def data_ptr(self, index: int = 0) -> NDArray:
        return self.data.reshape(self.L5, -1)[index]

    def even_ptr(self, index: int) -> NDArray:
        assert self.full_field
        return self.data.reshape(self.L5, 2, -1)[index, 0]

    def odd_ptr(self, index: int) -> NDArray:
        assert self.full_field
        return self.data.reshape(self.L5, 2, -1)[index, 1]

    @property
    def data_ptrs(self) -> NDArray:
        return self.data.reshape(self.L5, -1)

    @property
    def even_ptrs(self) -> NDArray:
        assert self.full_field
        return self.data.reshape(self.L5, 2, -1)[:, 0]

    @property
    def odd_ptrs(self) -> NDArray:
        assert self.full_field
        return self.data.reshape(self.L5, 2, -1)[:, 1]

    def copy(self):
        return self.__class__(self.latt_info, self.L5, self.backup())

    def lexico(self):
        return lexico(self.getHost(), [1, 2, 3, 4, 5])

    def checksum(self) -> List[Tuple[int, int]]:
        return [self[index].checksum() for index in range(self.L5)]

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


class LatticeInt(FullField, ParityField):
    pass


class LatticeReal(FullField, ParityField):
    pass


class LatticeComplex(FullField, ParityField):
    pass


class LatticeLink(FullField, ParityField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info, value)
        if value is None:
            if self.backend == "numpy":
                self.data[:] = numpy.identity(latt_info.Nc)
            elif self.backend == "cupy":
                import cupy

                self.data[:] = cupy.identity(latt_info.Nc)
            elif self.backend == "torch":
                import torch

                self.data[:] = torch.eye(latt_info.Nc)

    @property
    def __field_class__(self):
        return LatticeLink

    def pack(self, x: "LatticeFermion"):
        for color in range(self.latt_info.Nc):
            x.data[:, :, :, :, :, color, :] = self.data[:, :, :, :, :, :, color]

    def unpack(self, x: "LatticeFermion"):
        for color in range(self.latt_info.Nc):
            self.data[:, :, :, :, :, :, color] = x.data[:, :, :, :, :, color, :]


class LatticeGauge(MultiField, LatticeLink):
    def __init__(self, latt_info: LatticeInfo, L5: Union[int, Any] = Nd, value=None) -> None:
        """`L5` can be `value` here"""
        if not isinstance(L5, int):
            value = L5
            L5 = latt_info.Nd
        super().__init__(latt_info, L5, value)
        if value is None:
            if self.backend == "numpy":
                self.data[:] = numpy.identity(latt_info.Nc)
            elif self.backend == "cupy":
                import cupy

                self.data[:] = cupy.identity(latt_info.Nc)
            elif self.backend == "torch":
                import torch

                self.data[:] = torch.eye(latt_info.Nc)

    @classmethod
    def load(cls, filename: str, *, check: bool = True) -> "LatticeGauge":
        return super().load(filename, ["X", "Y", "Z", "T"], check=check)

    def save(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().save(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def append(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().append(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def update(self, filename: str, *, annotation: str = "", check: bool = True):
        super().update(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check)

    @property
    def gauge_dirac(self):
        if not hasattr(self, "_gauge_dirac"):
            from pyquda.dirac import GaugeDirac

            self._gauge_dirac = GaugeDirac(self.latt_info)
        return self._gauge_dirac

    @property
    def pure_gauge(self):
        return self.gauge_dirac

    def setAntiPeriodicT(self):
        if self.latt_info.gt == self.latt_info.Gt - 1:
            self.data[self.latt_info.Nd - 1, :, self.latt_info.Lt - 1] *= -1

    def setAnisotropy(self, anisotropy: float):
        self.data[: self.latt_info.Nd - 1] /= anisotropy

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

    def apeSmear(
        self,
        n_steps: int,
        alpha: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.apeSmear(n_steps, alpha, dir_ignore, compute_plaquette, compute_qcharge)
        self._gauge_dirac.saveSmearedGauge(self)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()

    def stoutSmear(
        self,
        n_steps: int,
        rho: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.stoutSmear(n_steps, rho, dir_ignore, compute_plaquette, compute_qcharge)
        self._gauge_dirac.saveSmearedGauge(self)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()

    def hypSmear(
        self,
        n_steps: int,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore, compute_plaquette, compute_qcharge)
        self._gauge_dirac.saveSmearedGauge(self)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()

    def wilsonFlow(
        self,
        n_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.wilsonFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
        energy = [self._gauge_dirac.obs_param.energy]
        for step in range(1, n_steps):
            self._gauge_dirac.wilsonFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
            energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.saveSmearedGauge(self)  # Save before the last step
        self._gauge_dirac.wilsonFlow(1, epsilon, n_steps * epsilon, True, compute_plaquette, compute_qcharge)
        energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return energy

    def wilsonFlowScale(
        self,
        max_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.wilsonFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
        t2E, tdt2E = 0, 0
        t0, w0 = 0, 0
        for step in range(1, max_steps + 1):
            if t2E >= 0.3 and tdt2E >= 0.3:
                break
            self._gauge_dirac.wilsonFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
            t2E_old, t2E = t2E, (step * epsilon) ** 2 * self._gauge_dirac.obs_param.energy[0]
            tdt2E_old, tdt2E = tdt2E, (step - 0.5) * (t2E - t2E_old)
            if t0 == 0 and t2E >= 0.3:
                t0 = (step - (t2E - 0.3) / (t2E - t2E_old)) * epsilon
            if w0 == 0 and tdt2E >= 0.3:
                w0 = ((step - 0.5 - (tdt2E - 0.3) / (tdt2E - tdt2E_old)) * epsilon) ** 0.5
            getLogger().info(f"t2E({step * epsilon:.3f})={t2E}, tdt2E({(step - 0.5) * epsilon:.3f})={tdt2E}")
        else:
            getLogger().error(
                f"Wilson flow scale doesn't exceed 0.3 at max_steps*epsilon={max_steps * epsilon}", RuntimeError
            )
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return t0, w0

    def symanzikFlow(
        self,
        n_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.symanzikFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
        energy = [self._gauge_dirac.obs_param.energy]
        for step in range(1, n_steps):
            self._gauge_dirac.symanzikFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
            energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.saveSmearedGauge(self)  # Save before the last step
        self._gauge_dirac.symanzikFlow(1, epsilon, n_steps * epsilon, True, compute_plaquette, compute_qcharge)
        energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return energy

    def symanzikFlowScale(
        self,
        max_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.symanzikFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
        t2E, tdt2E = 0, 0
        t0, w0 = 0, 0
        for step in range(1, max_steps + 1):
            if t2E >= 0.3 and tdt2E >= 0.3:
                break
            self._gauge_dirac.symanzikFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
            t2E_old, t2E = t2E, (step * epsilon) ** 2 * self._gauge_dirac.obs_param.energy[0]
            tdt2E_old, tdt2E = tdt2E, (step - 0.5) * (t2E - t2E_old)
            if t0 == 0 and t2E >= 0.3:
                t0 = (step - (t2E - 0.3) / (t2E - t2E_old)) * epsilon
            if w0 == 0 and tdt2E >= 0.3:
                w0 = ((step - 0.5 - (tdt2E - 0.3) / (tdt2E - tdt2E_old)) * epsilon) ** 0.5
            getLogger().info(f"t2E({step * epsilon:.3f})={t2E}, tdt2E({(step - 0.5) * epsilon:.3f})={tdt2E}")
        else:
            getLogger().error(
                f"Symanzik flow scale doesn't exceed 0.3 at max_steps*epsilon={max_steps * epsilon}", RuntimeError
            )
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return t0, w0

    def smearAPE(
        self,
        n_steps: int,
        factor: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        """A variant of apeSmear() to match Chroma"""
        dimAPE = 3 if dir_ignore >= 0 and dir_ignore <= 3 else 4
        self.apeSmear(n_steps, (dimAPE - 1) / (dimAPE - 1 + factor / 2), dir_ignore, compute_plaquette, compute_qcharge)

    def smearSTOUT(
        self,
        n_steps: int,
        rho: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.stoutSmear(n_steps, rho, dir_ignore, compute_plaquette, compute_qcharge)

    def smearHYP(
        self,
        n_steps: int,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore, compute_plaquette, compute_qcharge)

    def flowWilson(self, n_steps: int, time: float, compute_plaquette: bool = False, compute_qcharge: bool = True):
        return self.wilsonFlow(n_steps, time / n_steps, compute_plaquette, compute_qcharge)

    def flowWilsonScale(self, epsilon: float, compute_plaquette: bool = False, compute_qcharge: bool = True):
        return self.wilsonFlowScale(100000, epsilon, compute_plaquette, compute_qcharge)

    def flowSymanzik(self, n_steps: int, time: float, compute_plaquette: bool = False, compute_qcharge: bool = True):
        return self.symanzikFlow(n_steps, time / n_steps, compute_plaquette, compute_qcharge)

    def flowSymanzikScale(self, epsilon: float, compute_plaquette: bool = False, compute_qcharge: bool = True):
        return self.symanzikFlowScale(100000, epsilon, compute_plaquette, compute_qcharge)

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

        Parameters
        ----------
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


class LatticeMom(MultiField, FullField, ParityField):
    def __init__(self, latt_info: LatticeInfo, L5: Union[int, Any] = 4, value=None) -> None:
        """`L5` can be `value` here"""
        if not isinstance(L5, int):
            value = L5
            L5 = latt_info.Nd
        super().__init__(latt_info, L5, value)
        self._gauge_dirac = None

    @classmethod
    def load(cls, filename: str, *, check: bool = True) -> "LatticeMom":
        return super().load(filename, ["X", "Y", "Z", "T"], check=check)

    def save(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().save(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def append(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().append(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def update(self, filename: str, *, annotation: str = "", check: bool = True):
        super().update(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check)

    @property
    def gauge_dirac(self):
        if self._gauge_dirac is None:
            from pyquda.dirac import GaugeDirac

            self._gauge_dirac = GaugeDirac(self.latt_info)
        return self._gauge_dirac

    def gauss(self, seed: int, sigma: float):
        self.gauge_dirac.loadMom(self)
        self._gauge_dirac.gaussMom(seed, sigma)
        self._gauge_dirac.saveFreeMom(self)


class LatticeClover(FullField, ParityField):
    pass


class HalfLatticeFermion(ParityField):
    @property
    def __field_class__(self):
        return HalfLatticeFermion


class LatticeFermion(FullField, HalfLatticeFermion):
    @property
    def __field_class__(self):
        return LatticeFermion


class MultiHalfLatticeFermion(MultiField, HalfLatticeFermion):
    pass


class MultiLatticeFermion(MultiField, LatticeFermion):
    pass


class LatticePropagator(FullField, ParityField):
    def setFermion(self, fermion: LatticeFermion, spin: int, color: int):
        self.data[:, :, :, :, :, :, spin, :, color] = fermion.data

    def getFermion(self, spin: int, color: int):
        return LatticeFermion(self.latt_info, self.data[:, :, :, :, :, :, spin, :, color])


class HalfLatticeStaggeredFermion(ParityField):
    @property
    def __field_class__(self):
        return HalfLatticeStaggeredFermion


class MultiHalfLatticeStaggeredFermion(MultiField, HalfLatticeStaggeredFermion):
    pass


class LatticeStaggeredFermion(FullField, HalfLatticeStaggeredFermion):
    @property
    def __field_class__(self):
        return LatticeStaggeredFermion


class MultiLatticeStaggeredFermion(MultiField, LatticeStaggeredFermion):
    pass


class LatticeStaggeredPropagator(FullField, ParityField):
    def setFermion(self, fermion: LatticeStaggeredFermion, color: int):
        self.data[:, :, :, :, :, :, color] = fermion.data

    def getFermion(self, color: int) -> LatticeStaggeredFermion:
        return LatticeStaggeredFermion(self.latt_info, self.data[:, :, :, :, :, :, color])
