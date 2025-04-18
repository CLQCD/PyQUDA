from abc import abstractmethod
from math import prod
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

    def __init__(
        self, latt_size: List[int], t_boundary: Literal[1, -1] = 1, anisotropy: float = 1.0, Ns: int = 4, Nc: int = 3
    ) -> None:
        self.Nd, self.Ns, self.Nc = len(latt_size), Ns, Nc
        self._checkLattice(latt_size)
        self._setLattice(latt_size, t_boundary, anisotropy)

    def _checkLattice(self, latt_size: List[int]):
        grid_size = getGridSize()
        assert len(latt_size) == len(grid_size), "lattice and grid must have the same dimension"
        if not all([(GL % (2 * G) == 0 or GL * G == 1) for GL, G in zip(latt_size, grid_size)]):
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

        sublatt_size = [GL // G for GL, G in zip(latt_size, self.grid_size)]
        if self.Nd == 4:
            self.Gx, self.Gy, self.Gz, self.Gt = self.grid_size
            self.gx, self.gy, self.gz, self.gt = self.grid_coord
            self.GLx, self.GLy, self.GLz, self.GLt = latt_size
            self.Lx, self.Ly, self.Lz, self.Lt = sublatt_size
        else:
            self.Gt = self.grid_size[-1]
            self.gt = self.grid_coord[-1]
            self.GLt = latt_size[-1]
            self.Lt = sublatt_size[-1]

        self.global_size = latt_size
        self.global_volume = prod(latt_size)
        self.size = sublatt_size
        self.volume = prod(sublatt_size)
        self.ga_pad = self.volume // min(sublatt_size) // 2

        self.t_boundary = t_boundary
        self.anisotropy = anisotropy


class GeneralInfo:
    def __init__(self, latt_size: List[int], Ns: int = 4, Nc: int = 3) -> None:
        self._checkLattice(latt_size)
        self._setLattice(latt_size)
        self.Nd = len(latt_size)
        self.Ns = Ns
        self.Nc = Nc

    def _checkLattice(self, latt_size: List[int]):
        grid_size = getGridSize()
        assert len(latt_size) == len(grid_size), "lattice size and grid size must have the same dimension"
        if not all([(GL % G == 0) for GL, G in zip(latt_size, grid_size)]):
            getLogger().critical("lattice size must be divisible by gird size", ValueError)

    def _setLattice(self, latt_size: List[int]):
        self.mpi_comm = getMPIComm()
        self.mpi_size = getMPISize()
        self.mpi_rank = getMPIRank()
        self.grid_size = getGridSize()
        self.grid_coord = getGridCoord()

        sublatt_size = [GL // G for GL, G in zip(latt_size, self.grid_size)]

        self.global_size = latt_size
        self.global_volume = prod(latt_size)
        self.size = sublatt_size
        self.volume = prod(self.size)


def lexico(data: numpy.ndarray, axes: List[int], dtype=None):
    assert len(axes) == 5
    shape = data.shape
    Np, Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    assert Np == 2
    Lx *= 2
    Npre = prod(shape[: axes[0]])
    Nsuf = prod(shape[axes[-1] + 1 :])
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
    assert len(axes) == 4
    shape = data.shape
    Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    Npre = prod(shape[: axes[0]])
    Nsuf = prod(shape[axes[-1] + 1 :])
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
    elif field in ["SpinVector"]:
        return [Ns], f"<c{2 * float_nbytes}"
    elif field in ["SpinMatrix"]:
        return [Ns, Ns], f"<c{2 * float_nbytes}"
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
        self.field_size = prod(field_shape)
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

    def __add__(self, rhs):
        if not self.__class__ == rhs.__class__:
            return NotImplemented
        assert self.location == rhs.location
        return self.__class__(self.latt_info, self.data + rhs.data)

    def __sub__(self, rhs):
        if not self.__class__ == rhs.__class__:
            return NotImplemented
        assert self.location == rhs.location
        return self.__class__(self.latt_info, self.data - rhs.data)

    def __mul__(self, rhs):
        return self.__class__(self.latt_info, self.data * rhs)

    def __rmul__(self, lhs):
        return self.__class__(self.latt_info, lhs * self.data)

    def __truediv__(self, rhs):
        return self.__class__(self.latt_info, self.data / rhs)

    def __neg__(self):
        return self.__class__(self.latt_info, -self.data)

    def __iadd__(self, rhs):
        if not self.__class__ == rhs.__class__:
            return NotImplemented
        assert self.location == rhs.location
        self._data += rhs.data
        return self

    def __isub__(self, rhs):
        if not self.__class__ == rhs.__class__:
            return NotImplemented
        assert self.location == rhs.location
        self._data -= rhs.data
        return self

    def __imul__(self, rhs):
        self._data *= rhs
        return self

    def __itruediv__(self, rhs):
        self._data /= rhs
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
        latt_size = self.latt_info.size
        self.lattice_shape = (
            [2, *latt_size[1:][::-1], latt_size[0] // 2]
            if self.full_field
            else [*latt_size[1:][::-1], latt_size[0] // 2]
        )
        if self.L5 is None:
            return (*self.lattice_shape, *self.field_shape)
        else:
            return (self.L5, *self.lattice_shape, *self.field_shape)

    def timeslice(self, start: int, stop: int = None, step: int = None, return_field: bool = True):
        Lt = self.latt_info.size[0]
        gt = self.latt_info.grid_size[0]
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
                x.data[:, :, start:stop:step] = self.data[:, :, start:stop:step]
            elif self.full_field or self.L5 is not None:
                x.data[:, start:stop:step] = self.data[:, start:stop:step]
            else:
                x.data[start:stop:step] = self.data[start:stop:step]
        else:
            if self.full_field and self.L5 is not None:
                x = self.data[:, :, start:stop:step]
            elif self.full_field or self.L5 is not None:
                x = self.data[:, start:stop:step]
            else:
                x = self.data[start:stop:step]
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
        return lexico(self.getHost(), list(range(0, self.latt_info.Nd + 1)), dtype)

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
            retval = cls(latt_info, evenodd(value, list(range(0, latt_info.Nd))))
        else:
            retval = cls(
                latt_info, len(label), numpy.asarray([evenodd(data, list(range(0, latt_info.Nd))) for data in value])
            )
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

    def lexico(self, dtype=None):
        return lexico(self.getHost(), list(range(1, self.latt_info.Nd + 2)), dtype)

    def checksum(self) -> List[Tuple[int, int]]:
        return [self[index].checksum() for index in range(self.L5)]

    def __add__(self, rhs):
        if not self.__class__ == rhs.__class__:
            return NotImplemented
        assert self.location == rhs.location
        return self.__class__(self.latt_info, self.L5, self.data + rhs.data)

    def __sub__(self, rhs):
        if not self.__class__ == rhs.__class__:
            return NotImplemented
        assert self.location == rhs.location
        return self.__class__(self.latt_info, self.L5, self.data - rhs.data)

    def __mul__(self, rhs):
        return self.__class__(self.latt_info, self.L5, self.data * rhs)

    def __rmul__(self, lhs):
        return self.__class__(self.latt_info, self.L5, lhs * self.data)

    def __truediv__(self, rhs):
        return self.__class__(self.latt_info, self.L5, self.data / rhs)

    def __neg__(self):
        return self.__class__(self.latt_info, self.L5, -self.data)


class LatticeInt(FullField, ParityField):
    pass


class LatticeReal(FullField, ParityField):
    pass


class LatticeComplex(FullField, ParityField):
    pass


class HalfLatticeSpinColorVector(ParityField):
    @property
    def __field_class__(self):
        return HalfLatticeSpinColorVector


class LatticeSpinColorVector(FullField, HalfLatticeSpinColorVector):
    @property
    def __field_class__(self):
        return LatticeSpinColorVector


class LatticeSpinColorMatrix(FullField, ParityField):
    @property
    def __field_class__(self):
        return LatticeColorMatrix


class HalfLatticeColorVector(ParityField):
    @property
    def __field_class__(self):
        return HalfLatticeColorVector


class LatticeColorVector(FullField, HalfLatticeColorVector):
    @property
    def __field_class__(self):
        return LatticeColorVector


class LatticeColorMatrix(FullField, ParityField):
    @property
    def __field_class__(self):
        return LatticeColorMatrix


class LatticeSpinMatrix(FullField, ParityField):
    @property
    def __field_class__(self):
        return LatticeSpinMatrix


class LatticeLink(LatticeColorMatrix):
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


class LatticeGauge(MultiField, LatticeLink):
    def __init__(self, latt_info: LatticeInfo, L5: int, value=None) -> None:
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

    def setAntiPeriodicT(self):
        if self.latt_info.gt == self.latt_info.Gt - 1:
            self.data[-1, :, -1] *= -1

    def setAnisotropy(self, anisotropy: float):
        self.data[:-1] /= anisotropy


class LatticeMom(MultiField, FullField, ParityField):
    pass


class LatticeClover(FullField, ParityField):
    pass


class HalfLatticeFermion(HalfLatticeSpinColorVector):
    @property
    def __field_class__(self):
        return HalfLatticeFermion


class MultiHalfLatticeFermion(MultiField, HalfLatticeFermion):
    pass


class LatticeFermion(LatticeSpinColorVector):
    @property
    def __field_class__(self):
        return LatticeFermion


class MultiLatticeFermion(MultiField, LatticeFermion):
    pass


class LatticePropagator(LatticeSpinColorMatrix):
    def setFermion(self, fermion: LatticeFermion, spin: int, color: int):
        self.data[:, :, :, :, :, :, spin, :, color] = fermion.data

    def getFermion(self, spin: int, color: int):
        return LatticeFermion(self.latt_info, self.data[:, :, :, :, :, :, spin, :, color])


class HalfLatticeStaggeredFermion(HalfLatticeColorVector):
    @property
    def __field_class__(self):
        return HalfLatticeStaggeredFermion


class MultiHalfLatticeStaggeredFermion(MultiField, HalfLatticeStaggeredFermion):
    pass


class LatticeStaggeredFermion(LatticeColorVector):
    @property
    def __field_class__(self):
        return LatticeStaggeredFermion


class MultiLatticeStaggeredFermion(MultiField, LatticeStaggeredFermion):
    pass


class LatticeStaggeredPropagator(LatticeColorMatrix):
    def setFermion(self, fermion: LatticeStaggeredFermion, color: int):
        self.data[:, :, :, :, :, :, color] = fermion.data

    def getFermion(self, color: int):
        return LatticeStaggeredFermion(self.latt_info, self.data[:, :, :, :, :, :, color])
