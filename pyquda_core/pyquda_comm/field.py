from abc import abstractmethod
from math import prod
from os import path
from time import perf_counter
from typing import Any, List, Literal, Sequence, Tuple, Union

import numpy
from numpy.lib.format import dtype_to_descr, read_magic, read_array_header_1_0, write_array_header_1_0
from numpy.typing import NDArray

from . import (
    getRankFromCoord,
    getLogger,
    getSublatticeSize,
    getMPIComm,
    getMPISize,
    getMPIRank,
    getGridSize,
    getGridCoord,
    getCUDABackend,
    readMPIFile,
    writeMPIFile,
    fieldFFT,
)
from .array import (
    BackendType,
    arrayDType,
    arrayHost,
    arrayHostCopy,
    arrayDevice,
    arrayCopy,
    arrayIsContiguous,
    arrayContiguous,
    arrayNorm2,
    arrayIdentity,
    arrayZeros,
)


class LatticeInfo:
    def __init__(
        self, latt_size: List[int], t_boundary: Literal[1, -1] = 1, anisotropy: float = 1.0, Ns: int = 4, Nc: int = 3
    ) -> None:
        self.Nd = len(latt_size)
        self.Ns = Ns
        self.Nc = Nc
        self._setLattice(latt_size, t_boundary, anisotropy)

    def _setLattice(self, latt_size: List[int], t_boundary: Literal[1, -1], anisotropy: float):
        self.mpi_comm = getMPIComm()
        self.mpi_size = getMPISize()
        self.mpi_rank = getMPIRank()
        self.grid_size = getGridSize()
        self.grid_coord = getGridCoord()

        sublatt_size = getSublatticeSize(latt_size)
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

    def _setEvenOdd(self):
        if hasattr(self, "even") and hasattr(self, "odd"):
            return
        eo = numpy.sum(numpy.indices(self.size[::-1], "<i4"), axis=0).reshape(-1) % 2
        self.even = eo == 0
        self.odd = eo == 1

    def lexico(self, data: numpy.ndarray, multi: bool, backend: BackendType = "numpy"):
        self._setEvenOdd()
        shape = data.shape
        if multi:
            L5 = shape[0]
            assert shape[1] == 2
            sublatt_size = list(shape[2 : self.Nd + 2][::-1])
            field_shape = list(shape[self.Nd + 2 :])
        else:
            L5 = 1
            assert shape[0] == 2
            sublatt_size = list(shape[1 : self.Nd + 1][::-1])
            field_shape = list(shape[self.Nd + 1 :])
        sublatt_size[0] *= 2
        assert sublatt_size == self.size
        data_evenodd = data.reshape(L5, 2, self.volume // 2, prod(field_shape))
        data_lexico = arrayZeros((L5, self.volume, prod(field_shape)), data.dtype, backend)
        data_lexico[:, self.even] = data_evenodd[:, 0]
        data_lexico[:, self.odd] = data_evenodd[:, 1]
        if multi:
            return data_lexico.reshape(L5, *sublatt_size[::-1], *field_shape)
        else:
            return data_lexico.reshape(*sublatt_size[::-1], *field_shape)

    def evenodd(self, data: numpy.ndarray, multi: bool, backend: BackendType = "numpy"):
        self._setEvenOdd()
        shape = data.shape
        if multi:
            L5 = shape[0]
            sublatt_size = list(shape[1 : self.Nd + 1][::-1])
            field_shape = list(shape[self.Nd + 1 :])
        else:
            L5 = 1
            sublatt_size = list(shape[0 : self.Nd + 0][::-1])
            field_shape = list(shape[self.Nd + 0 :])
        assert sublatt_size == self.size
        sublatt_size[0] //= 2
        data_lexico = data.reshape(L5, self.volume, prod(field_shape))
        data_evenodd = arrayZeros((L5, 2, self.volume // 2, prod(field_shape)), data.dtype, backend)
        data_evenodd[:, 0] = data_lexico[:, self.even]
        data_evenodd[:, 1] = data_lexico[:, self.odd]
        if multi:
            return data_evenodd.reshape(L5, 2, *sublatt_size[::-1], *field_shape)
        else:
            return data_evenodd.reshape(2, *sublatt_size[::-1], *field_shape)


class GeneralInfo:
    def __init__(self, latt_size: List[int], Ns: int = 4, Nc: int = 3) -> None:
        self.Nd = len(latt_size)
        self.Ns = Ns
        self.Nc = Nc
        self._setLattice(latt_size)

    def _setLattice(self, latt_size: List[int]):
        self.mpi_comm = getMPIComm()
        self.mpi_size = getMPISize()
        self.mpi_rank = getMPIRank()
        self.grid_size = getGridSize()
        self.grid_coord = getGridCoord()

        sublatt_size = getSublatticeSize(latt_size, False)

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


def read_array_header(filename: str) -> Tuple[Tuple[int, ...], str, int]:
    with open(filename, "rb") as f:
        assert read_magic(f) == (1, 0)
        shape, fortran_order, dtype = read_array_header_1_0(f)
        dtype = dtype_to_descr(dtype)
        assert not fortran_order
        offset = f.tell()
    return shape, dtype, offset


def write_array_header(filename: str, shape: Tuple[int, ...], dtype: str):
    if getMPIRank() == 0:
        with open(filename, "wb") as f:
            write_array_header_1_0(f, {"shape": shape, "fortran_order": False, "descr": dtype})
    getMPIComm().Barrier()


def _field_spin_color_dtype(
    field: str, shape: Sequence[int], use_fp32: bool
) -> Tuple[Union[int, None], Union[int, None], str]:
    float_nbytes = 4 if use_fp32 else 8
    if field in ["Int"]:
        () = shape
        return None, None, "<i4"
    elif field in ["Real"]:
        return None, None, f"<f{float_nbytes}"
    elif field in ["Complex"]:
        return None, None, f"<c{2 * float_nbytes}"
    elif field in ["SpinColorVector"]:
        Ns, Nc = shape
        return Ns, Nc, f"<c{2 * float_nbytes}"
    elif field in ["SpinColorMatrix"]:
        Ns, Ns_, Nc, Nc_ = shape
        assert Ns == Ns_ and Nc == Nc_
        return Ns, Nc, f"<c{2 * float_nbytes}"
    elif field in ["SpinVector"]:
        (Ns,) = shape
        return Ns, None, f"<c{2 * float_nbytes}"
    elif field in ["SpinMatrix"]:
        Ns, Ns_ = shape
        assert Ns == Ns_
        return Ns, None, f"<c{2 * float_nbytes}"
    elif field in ["ColorVector"]:
        (Nc,) = shape
        return None, Nc, f"<c{2 * float_nbytes}"
    elif field in ["ColorMatrix"]:
        Nc, Nc_ = shape
        assert Nc == Nc_
        return None, Nc, f"<c{2 * float_nbytes}"
    else:
        getLogger().critical(f"Unknown field type: {field}", ValueError)


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
        self.backend: BackendType = getCUDABackend()
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

    @classmethod
    def loadNPY(cls, filename: str):
        assert issubclass(cls, (GeneralField, FullField))
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        grid_size = getGridSize()
        Nd = len(grid_size)
        shape, dtype, offset = read_array_header(filename)
        if not issubclass(cls, MultiField):
            latt_size = list(shape[:Nd][::-1])
            field_shape = list(shape[Nd:])
            sublatt_size = [GL // G for GL, G in zip(latt_size, grid_size)]
            value = readMPIFile(
                filename, dtype, offset, [*sublatt_size[::-1], *field_shape], list(range(Nd - 1, -1, -1))
            )
            gbytes += value.nbytes / 1024**3
        else:
            L5 = shape[0]
            latt_size = list(shape[1 : Nd + 1][::-1])
            field_shape = list(shape[Nd + 1 :])
            sublatt_size = [GL // G for GL, G in zip(latt_size, grid_size)]
            value = readMPIFile(
                filename, dtype, offset, [L5, *sublatt_size[::-1], *field_shape], list(range(Nd, 0, -1))
            )
            gbytes += value.nbytes / 1024**3
        Ns, Nc, field_dtype = _field_spin_color_dtype(cls._field(), field_shape, False)
        value = value.astype(field_dtype)
        if issubclass(cls, GeneralField):
            latt_info = GeneralInfo(latt_size)
            if Ns is not None:
                latt_info.Ns = Ns
            if Nc is not None:
                latt_info.Nc = Nc
            if not issubclass(cls, MultiField):
                retval = cls(latt_info, value)
            else:
                retval = cls(latt_info, L5, value)
        elif issubclass(cls, FullField):
            latt_info = LatticeInfo(latt_size)
            if Ns is not None:
                latt_info.Ns = Ns
            if Nc is not None:
                latt_info.Nc = Nc
            if not issubclass(cls, MultiField):
                retval = cls(latt_info, latt_info.evenodd(value, False))
            else:
                retval = cls(latt_info, L5, latt_info.evenodd(value, True))
        secs = perf_counter() - s
        getLogger().debug(f"Loaded {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")
        return retval

    def saveNPY(self, filename: str, *, use_fp32: bool = False):
        assert isinstance(self, (GeneralField, FullField))
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        if not filename.endswith(".npy"):
            filename += ".npy"
        field = self.lexico()
        _, _, dtype = _field_spin_color_dtype(self._field(), self.field_shape, use_fp32)
        if self.L5 is None:
            write_array_header(filename, (*self.latt_info.global_size[::-1], *self.field_shape), dtype)
            _, _, offset = read_array_header(filename)
            shape = (*self.latt_info.size[::-1], *self.field_shape)
            axes = list(range(self.latt_info.Nd - 1, -1, -1))
            writeMPIFile(filename, dtype, offset, shape, axes, field.astype(dtype))
            gbytes += field.nbytes / 1024**3
        else:
            write_array_header(filename, (self.L5, *self.latt_info.global_size[::-1], *self.field_shape), dtype)
            _, _, offset = read_array_header(filename)
            shape = (self.L5, *self.latt_info.size[::-1], *self.field_shape)
            axes = list(range(self.latt_info.Nd, 0, -1))
            writeMPIFile(filename, dtype, offset, shape, axes, field.astype(dtype))
            gbytes += field.nbytes / 1024**3
        secs = perf_counter() - s
        getLogger().debug(f"Saved {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")

    @classmethod
    def load(
        cls,
        filename: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        *,
        check: bool = True,
    ):
        from .hdf5 import File

        assert issubclass(cls, (GeneralField, FullField))
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        with File(filename, "r") as f:
            latt_size, Ns, Nc, value = f.load(cls._groupName(), label, check=check)
        if issubclass(cls, GeneralField):
            latt_info = GeneralInfo(latt_size)
            if Ns is not None:
                latt_info.Ns = Ns
            if Nc is not None:
                latt_info.Nc = Nc
            if not issubclass(cls, MultiField):
                retval = cls(latt_info, value)
            else:
                retval = cls(latt_info, len(label), value)
        elif issubclass(cls, FullField):
            latt_info = LatticeInfo(latt_size)
            if Ns is not None:
                latt_info.Ns = Ns
            if Nc is not None:
                latt_info.Nc = Nc
            if not issubclass(cls, MultiField):
                retval = cls(latt_info, latt_info.evenodd(value, False))
            else:
                retval = cls(latt_info, len(label), latt_info.evenodd(value, True))
        secs = perf_counter() - s
        getLogger().debug(f"Loaded {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")
        return retval

    def save(
        self,
        filename: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        *,
        annotation: str = "",
        check: bool = True,
        use_fp32: bool = False,
    ):
        from .hdf5 import File

        assert isinstance(self, (GeneralField, FullField))
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        if not filename.endswith(".h5") and not filename.endswith(".hdf5"):
            filename += ".h5"
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
        from .hdf5 import File

        assert isinstance(self, (GeneralField, FullField))
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
        from .hdf5 import File

        assert isinstance(self, (GeneralField, FullField))
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
        location = "numpy" if isinstance(value, numpy.ndarray) else self.backend
        self._data = arrayContiguous(value, location)

    @property
    def data_ptr(self) -> NDArray:
        return self.data.reshape(-1)

    @classmethod
    def _field(cls) -> str:
        group_name = cls._groupName()
        return group_name[group_name.index("Lattice") + len("Lattice") :]

    def _setField(self):
        field_shape, field_dtype = _field_shape_dtype(self._field(), self.latt_info.Ns, self.latt_info.Nc)
        self.field_shape = field_shape
        self.field_size = prod(field_shape)
        self.field_dtype = field_dtype
        self.shape = self._shape()
        self.dtype = arrayDType(field_dtype, self.backend)

    def _initData(self, value):
        self._setField()
        backend, value = (value, None) if isinstance(value, str) else (None, value)
        if value is None:
            location = backend if backend is not None else self.backend
            self.data = arrayZeros(self.shape, self.dtype, location)
        else:
            self.data = value.reshape(self.shape)

    @property
    def location(self) -> BackendType:
        if isinstance(self.data, numpy.ndarray):
            return "numpy"
        else:
            return self.backend

    def backup(self):
        return arrayCopy(self.data, self.location)

    def copy(self):
        return self.__class__(self.latt_info, self.backup())

    def toDevice(self):
        self.data = arrayDevice(self.data, self.backend)

    def toHost(self):
        self.data = arrayHost(self.data, self.location)

    def getHost(self):
        return arrayHostCopy(self.data, self.location)

    def norm2(self, all_reduce=True) -> float:
        norm2 = arrayNorm2(self.data, self.location)
        if all_reduce:
            return getMPIComm().allreduce(norm2)
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

    def lexico(self, force_numpy: bool = True):
        if force_numpy:
            return self.getHost()
        else:
            return self.backup()

    def fft(self):
        data = self.lexico()
        data_hat = fieldFFT(self.latt_info.size[::-1] + self.field_shape, data)
        return self.__class__(self.latt_info, arrayDevice(data_hat, self.location))

    def checksum(self) -> Tuple[int, int]:
        return checksum(self.latt_info, self.lexico().reshape(self.latt_info.volume, self.field_size).view("<u4"))

    def shift(self, n: int, mu: int):
        assert n >= 0
        assert 0 <= mu < 2 * self.latt_info.Nd
        Nd = self.latt_info.Nd
        direction = 1 if mu < Nd else -1
        mu = mu % Nd
        left_slice = [slice(None, None) for nu in range(Nd)]
        right_slice = [slice(None, None) for nu in range(Nd)]
        left = self.backup()
        right = self.data if abs(n) <= 1 else self.backup()
        rank = getMPIRank()
        coord = [g for g in getGridCoord()]
        coord[mu] = (getGridCoord()[mu] - direction) % getGridSize()[mu]
        dest = getRankFromCoord(coord)
        coord[mu] = (getGridCoord()[mu] + direction) % getGridSize()[mu]
        source = getRankFromCoord(coord)

        if n == 0:
            left, right = right, left
        while n > 0:
            left_slice[mu] = slice(-1, None) if direction == 1 else slice(None, 1)
            right_slice[mu] = slice(None, 1) if direction == 1 else slice(-1, None)
            sendbuf = right[tuple(right_slice[::-1])]
            if rank == source and rank == dest:
                pass
            else:
                sendbuf_host = arrayHostCopy(sendbuf, self.location)
                request = getMPIComm().Isend(sendbuf_host, dest)

            left_slice[mu] = slice(None, -1) if direction == 1 else slice(1, None)
            right_slice[mu] = slice(1, None) if direction == 1 else slice(None, -1)
            left[tuple(left_slice[::-1])] = right[tuple(right_slice[::-1])]

            left_slice[mu] = slice(-1, None) if direction == 1 else slice(None, 1)
            right_slice[mu] = slice(None, 1) if direction == 1 else slice(-1, None)
            if rank == source and rank == dest:
                recvbuf = sendbuf
            else:
                recvbuf_host = numpy.empty_like(sendbuf_host)
                getMPIComm().Recv(recvbuf_host, source)
                request.Wait()
                recvbuf = arrayDevice(recvbuf_host, self.location)
            left[tuple(left_slice[::-1])] = recvbuf

            n -= 1
            left, right = right, left

        return self.__class__(self.latt_info, right)


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

    # def timeslice(self, start: int, stop: int = None, step: int = None, return_field: bool = True):
    #     Lt = self.latt_info.size[0]
    #     gt = self.latt_info.grid_size[0]
    #     stop = (start + 1) if stop is None else stop
    #     step = 1 if step is None else step
    #     s = (start - gt * Lt) % step if start < gt * Lt and stop > gt * Lt else 0
    #     start = min(max(start - gt * Lt, 0), Lt) + s
    #     stop = min(max(stop - gt * Lt, 0), Lt)
    #     assert start <= stop and step > 0
    #     if return_field:
    #         if self.L5 is None:
    #             x = self.__class__(self.latt_info)
    #         else:
    #             x = self.__class__(self.latt_info, self.L5)
    #         if self.full_field and self.L5 is not None:
    #             x.data[:, :, start:stop:step] = self.data[:, :, start:stop:step]
    #         elif self.full_field or self.L5 is not None:
    #             x.data[:, start:stop:step] = self.data[:, start:stop:step]
    #         else:
    #             x.data[start:stop:step] = self.data[start:stop:step]
    #     else:
    #         if self.full_field and self.L5 is not None:
    #             x = self.data[:, :, start:stop:step]
    #         elif self.full_field or self.L5 is not None:
    #             x = self.data[:, start:stop:step]
    #         else:
    #             x = self.data[start:stop:step]
    #     return x


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

    def lexico(self, force_numpy: bool = True):
        if force_numpy:
            return self.latt_info.lexico(self.getHost(), self.L5 is not None)
        else:
            return self.latt_info.lexico(self.data, self.L5 is not None, self.location)

    def fft(self):
        data = self.lexico()
        data_hat = fieldFFT(self.latt_info.size[::-1] + self.field_shape, data)
        return self.__class__(self.latt_info, arrayDevice(self.latt_info.evenodd(data_hat, False), self.location))

    def checksum(self) -> Tuple[int, int]:
        return checksum(self.latt_info, self.lexico().reshape(self.latt_info.volume, self.field_size).view("<u4"))

    def shift(self, n: int, mu: int):
        assert n >= 0
        assert 0 <= mu < 2 * self.latt_info.Nd
        Nd = self.latt_info.Nd
        direction = 1 if mu < Nd else -1
        mu = mu % Nd
        left_slice = [slice(None, None) for nu in range(Nd)]
        right_slice = [slice(None, None) for nu in range(Nd)]
        left = self.backup()
        right = self.data if abs(n) <= 1 else self.backup()
        rank = getMPIRank()
        coord = [g for g in getGridCoord()]
        coord[mu] = (getGridCoord()[mu] - direction) % getGridSize()[mu]
        dest = getRankFromCoord(coord)
        coord[mu] = (getGridCoord()[mu] + direction) % getGridSize()[mu]
        source = getRankFromCoord(coord)

        if n == 0:
            left, right = right, left
        while n > 0:
            if mu == 0 and n == 1:
                left_flat = left.reshape(2, prod(self.latt_info.size[1:]), self.latt_info.size[0] // 2, -1)
                right_flat = right.reshape(2 * prod(self.latt_info.size[1:]), self.latt_info.size[0] // 2, -1)
                eo = numpy.sum(numpy.indices((2, *self.latt_info.size[1:][::-1])), axis=0).reshape(-1) % 2
                even = eo == 0
                odd = eo == 1
                if direction == 1:
                    sendbuf = right_flat[even, 0]
                    if rank == source and rank == dest:
                        pass
                    else:
                        sendbuf_host = arrayHostCopy(sendbuf, self.location)
                        request = getMPIComm().Isend(sendbuf_host, dest)

                    right_tmp = right_flat[odd].reshape(
                        2, prod(self.latt_info.size[1:]) // 2, self.latt_info.size[0] // 2, -1
                    )
                    left_flat[1, even.reshape(2, -1)[1]] = right_tmp[0]
                    left_flat[0, even.reshape(2, -1)[0]] = right_tmp[1]
                    right_tmp = right_flat[even, 1:].reshape(
                        2, prod(self.latt_info.size[1:]) // 2, self.latt_info.size[0] // 2 - 1, -1
                    )
                    left_flat[1, odd.reshape(2, -1)[1], :-1] = right_tmp[0]
                    left_flat[0, odd.reshape(2, -1)[0], :-1] = right_tmp[1]

                    if rank == source and rank == dest:
                        recvbuf = sendbuf
                    else:
                        recvbuf_host = numpy.empty_like(sendbuf_host)
                        getMPIComm().Recv(recvbuf_host, source)
                        request.Wait()
                        recvbuf = arrayDevice(recvbuf_host, self.location)
                    right_tmp = recvbuf.reshape(2, prod(self.latt_info.size[1:]) // 2, -1)
                    left_flat[1, odd.reshape(2, -1)[1], -1] = right_tmp[0]
                    left_flat[0, odd.reshape(2, -1)[0], -1] = right_tmp[1]
                else:
                    sendbuf = right_flat[odd, -1]
                    if rank == source and rank == dest:
                        pass
                    else:
                        sendbuf_host = arrayHostCopy(sendbuf, self.location)
                        request = getMPIComm().Isend(sendbuf_host, dest)

                    right_tmp = right_flat[even].reshape(
                        2, prod(self.latt_info.size[1:]) // 2, self.latt_info.size[0] // 2, -1
                    )
                    left_flat[1, odd.reshape(2, -1)[1]] = right_tmp[0]
                    left_flat[0, odd.reshape(2, -1)[0]] = right_tmp[1]
                    right_tmp = right_flat[odd, :-1].reshape(
                        2, prod(self.latt_info.size[1:]) // 2, self.latt_info.size[0] // 2 - 1, -1
                    )
                    left_flat[1, even.reshape(2, -1)[1], 1:] = right_tmp[0]
                    left_flat[0, even.reshape(2, -1)[0], 1:] = right_tmp[1]

                    if rank == source and rank == dest:
                        recvbuf = sendbuf
                    else:
                        recvbuf_host = numpy.empty_like(sendbuf_host)
                        getMPIComm().Recv(recvbuf_host, source)
                        request.Wait()
                        recvbuf = arrayDevice(recvbuf_host, self.location)
                    right_tmp = recvbuf.reshape(2, prod(self.latt_info.size[1:]) // 2, -1)
                    left_flat[1, even.reshape(2, -1)[1], 0] = right_tmp[0]
                    left_flat[0, even.reshape(2, -1)[0], 0] = right_tmp[1]

                n -= 1
                left, right = right, left
            else:
                left_slice[mu] = slice(-1, None) if direction == 1 else slice(None, 1)
                right_slice[mu] = slice(None, 1) if direction == 1 else slice(-1, None)
                sendbuf = right[(slice(0, 2),) + tuple(right_slice[::-1])]
                if rank == source and rank == dest:
                    pass
                else:
                    sendbuf_host = arrayHostCopy(sendbuf, self.location)
                    request = getMPIComm().Isend(sendbuf_host, dest)

                left_slice[mu] = slice(None, -1) if direction == 1 else slice(1, None)
                right_slice[mu] = slice(1, None) if direction == 1 else slice(None, -1)
                if mu == 0:
                    left[(0,) + tuple(left_slice[::-1])] = right[(0,) + tuple(right_slice[::-1])]
                    left[(1,) + tuple(left_slice[::-1])] = right[(1,) + tuple(right_slice[::-1])]
                else:
                    left[(0,) + tuple(left_slice[::-1])] = right[(1,) + tuple(right_slice[::-1])]
                    left[(1,) + tuple(left_slice[::-1])] = right[(0,) + tuple(right_slice[::-1])]

                left_slice[mu] = slice(-1, None) if direction == 1 else slice(None, 1)
                right_slice[mu] = slice(None, 1) if direction == 1 else slice(-1, None)
                if rank == source and rank == dest:
                    recvbuf = sendbuf
                else:
                    recvbuf_host = numpy.empty_like(sendbuf_host)
                    getMPIComm().Recv(recvbuf_host, source)
                    request.Wait()
                    recvbuf = arrayDevice(recvbuf_host, self.location)
                if mu == 0:
                    left[(0,) + tuple(left_slice[::-1])] = recvbuf[0]
                    left[(1,) + tuple(left_slice[::-1])] = recvbuf[1]
                else:
                    left[(0,) + tuple(left_slice[::-1])] = recvbuf[1]
                    left[(1,) + tuple(left_slice[::-1])] = recvbuf[0]

                if mu == 0:
                    n -= 2
                else:
                    n -= 1
                left, right = right, left

        return self.__class__(self.latt_info, right)


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
        location = "numpy" if isinstance(value, numpy.ndarray) else self.backend
        for index in range(self.L5):
            contiguous &= arrayIsContiguous(value[index], location)
        if contiguous:
            self._data = value
        else:
            self._data = arrayContiguous(value, location)

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

    def checksum(self) -> List[Tuple[int, int]]:
        return [self[index].checksum() for index in range(self.L5)]

    def shift(self, n: Sequence[int], mu: Sequence[int]):
        left = self.copy()
        for i in range(self.L5):
            left[i] = self[i].shift(n[i], mu[i])
        return left

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
            self.data[:] = arrayIdentity(latt_info.Nc, self.location)

    @property
    def __field_class__(self):
        return LatticeLink


class LatticeRotation(MultiField, LatticeLink):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info, 1, value)
        if value is None:
            self.data[:] = arrayIdentity(latt_info.Nc, self.location)


class LatticeGauge(MultiField, LatticeLink):
    def __init__(self, latt_info: LatticeInfo, L5: int, value=None) -> None:
        super().__init__(latt_info, L5, value)
        if value is None:
            self.data[:] = arrayIdentity(latt_info.Nc, self.location)

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
