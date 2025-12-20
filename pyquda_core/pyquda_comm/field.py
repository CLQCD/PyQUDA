from abc import abstractmethod
from math import prod
from os import path
from time import perf_counter
from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, Tuple, Type, TypeVar, Union, overload

Self = TypeVar("Self", bound="BaseField")
SelfMulti = TypeVar("SelfMulti", bound="MultiField")
Field = TypeVar("Field", bound="BaseField")

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
    getArrayBackend,
    readMPIFile,
    writeMPIFile,
)
from .array import (
    BackendType,
    arrayDType,
    arrayHost,
    arrayHostCopy,
    arrayDevice,
    arrayCopy,
    arrayIsContiguous,
    arrayAsContiguous,
    arrayLinalgNorm,
    arrayIdentity,
    arrayZeros,
    arrayEmpty,
)
from .pointer import Pointer, ndarrayPointer

_latt_even_odd: Dict[Tuple[int, ...], Tuple[NDArray[numpy.intp], NDArray[numpy.intp]]] = {}
_latt_coord: Dict[Tuple[int, ...], NDArray[numpy.int32]] = {}
_latt_sort_key: Dict[Tuple[int, ...], NDArray[numpy.int64]] = {}
_latt_lexico_coord: Dict[Tuple[int, ...], NDArray[numpy.int32]] = {}
_latt_lexico_sort_key: Dict[Tuple[int, ...], NDArray[numpy.int64]] = {}


class BaseInfo:
    def __init__(self, latt_size: Sequence[int], force_even: bool, Ns: int = 4, Nc: int = 3) -> None:
        self.Nd = len(latt_size)
        self.Ns = Ns
        self.Nc = Nc
        self._setLattice(latt_size, force_even)

    def _setLattice(self, latt_size: Sequence[int], force_even: bool):
        self.mpi_comm = getMPIComm()
        self.mpi_size = getMPISize()
        self.mpi_rank = getMPIRank()
        self.grid_size = getGridSize()
        self.grid_coord = getGridCoord()

        sublatt_size = getSublatticeSize(latt_size, force_even)
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

        self.global_size = [GL for GL in latt_size]
        self.global_volume = prod(latt_size)
        self.size = sublatt_size
        self.volume = prod(sublatt_size)
        self.ga_pad = self.volume // min(sublatt_size) // 2

    @abstractmethod
    def lexico(self, data: NDArray, multi: bool, backend: BackendType = "numpy") -> NDArray:
        getLogger().critical(
            f"{self.__class__.__name__}.lexico(data, multi, backend) not implemented", NotImplementedError
        )

    @abstractmethod
    def coordinate(self, mu: Optional[int] = None) -> NDArray:
        getLogger().critical(f"{self.__class__.__name__}.coordinate(mu) not implemented", NotImplementedError)

    @abstractmethod
    def sortSelect(self, select: NDArray[numpy.intp]) -> NDArray:
        getLogger().critical(f"{self.__class__.__name__}.lexico_index() not implemented", NotImplementedError)


class LatticeInfo(BaseInfo):
    def __init__(
        self, latt_size: List[int], t_boundary: Literal[1, -1] = 1, anisotropy: float = 1.0, Ns: int = 4, Nc: int = 3
    ) -> None:
        super().__init__(latt_size, True, Ns, Nc)
        self.t_boundary = t_boundary
        self.anisotropy = anisotropy

    def _setEvenOdd(self):
        global _latt_even_odd
        key = tuple(self.global_size)
        if key not in _latt_even_odd:
            eo = numpy.sum(numpy.indices(self.size[::-1], "<i4"), axis=0).reshape(-1) % 2
            _latt_even_odd[key] = (numpy.where(eo == 0)[0], numpy.where(eo == 1)[0])
        self._even, self._odd = _latt_even_odd[key]

    def lexico(self, data: NDArray, multi: bool, backend: BackendType = "numpy") -> NDArray:
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
        data_lexico = arrayEmpty((L5, self.volume, prod(field_shape)), data.dtype, backend)
        data_lexico[:, self._even] = data_evenodd[:, 0]
        data_lexico[:, self._odd] = data_evenodd[:, 1]
        if multi:
            return data_lexico.reshape(L5, *sublatt_size[::-1], *field_shape)
        else:
            return data_lexico.reshape(*sublatt_size[::-1], *field_shape)

    def evenodd(self, data: NDArray, multi: bool, backend: BackendType = "numpy") -> NDArray:
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
        data_evenodd = arrayEmpty((L5, 2, self.volume // 2, prod(field_shape)), data.dtype, backend)
        data_evenodd[:, 0] = data_lexico[:, self._even]
        data_evenodd[:, 1] = data_lexico[:, self._odd]
        if multi:
            return data_evenodd.reshape(L5, 2, *sublatt_size[::-1], *field_shape)
        else:
            return data_evenodd.reshape(2, *sublatt_size[::-1], *field_shape)

    def coordinate(self, mu: Optional[int] = None):
        global _latt_coord
        key = tuple(self.global_size)
        if key not in _latt_coord:
            _latt_coord[key] = self.evenodd(numpy.indices(self.size[::-1], "<i4")[::-1], True)
            for i in range(self.Nd):
                _latt_coord[key][i] += self.grid_coord[i] * self.size[i]
        if mu is None:
            return _latt_coord[key]
        else:
            assert 0 <= mu < self.Nd
            return _latt_coord[key][mu]

    def sortSelect(self, select: NDArray[numpy.intp]) -> NDArray:
        global _latt_sort_key
        key = tuple(self.global_size)
        if key not in _latt_sort_key:
            _latt_sort_key[key] = numpy.array(self.coordinate(0), "<i8")
            for mu in range(1, self.Nd):
                _latt_sort_key[key] *= self.global_size[mu]
                _latt_sort_key[key] += self.coordinate(mu)
        return select[numpy.argsort(_latt_sort_key[key].reshape(-1)[select])]


class LexicoInfo(BaseInfo):
    def __init__(self, latt_size: Sequence[int], Ns: int = 4, Nc: int = 3) -> None:
        super().__init__(latt_size, False, Ns, Nc)

    def lexico(self, data: NDArray, multi: bool, backend: BackendType = "numpy") -> NDArray:
        return arrayCopy(data, backend)

    def coordinate(self, mu: Optional[int] = None):
        global _latt_lexico_coord
        key = tuple(self.global_size)
        if key not in _latt_lexico_coord:
            _latt_lexico_coord[key] = numpy.indices(self.size[::-1], "<i4")[::-1]
            for i in range(self.Nd):
                _latt_lexico_coord[key][i] += self.grid_coord[i] * self.size[i]
        if mu is None:
            return _latt_lexico_coord[key]
        else:
            assert 0 <= mu < self.Nd
            return _latt_lexico_coord[key][mu]

    def sortSelect(self, select: NDArray[numpy.intp]) -> NDArray:
        global _latt_lexico_sort_key
        key = tuple(self.global_size)
        if key not in _latt_lexico_sort_key:
            _latt_lexico_sort_key[key] = numpy.array(self.coordinate(0), "<i8")
            for mu in range(1, self.Nd):
                _latt_lexico_sort_key[key] *= self.global_size[mu]
                _latt_lexico_sort_key[key] += self.coordinate(mu).reshape(-1)
        return select[numpy.argsort(_latt_lexico_sort_key[key][select])]


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
) -> Tuple[Optional[int], Optional[int], str]:
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


def _field_shape_dtype(field: str, Ns: int, Nc: int, use_fp32: bool = False) -> Tuple[List[int], str]:
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
    def __init__(self, latt_info: BaseInfo, *args, **kwargs) -> None:
        self.latt_info = latt_info
        self._data: NDArray = numpy.empty((0, 0), "<c16")
        self.backend: BackendType = getArrayBackend()
        self.L5: int = 0

    @property
    def location(self) -> BackendType:
        if isinstance(self.data, numpy.ndarray):
            return "numpy"
        else:
            return self.backend

    @abstractmethod
    def _latticeShape(self) -> List[int]:
        getLogger().critical(f"{self.__class__.__name__}._latticeShape() not implemented", NotImplementedError)

    @classmethod
    def _groupName(cls):
        return (
            cls.__name__.replace("Multi", "")
            .replace("Lexico", "")
            .replace("Link", "ColorMatrix")
            .replace("Gauge", "ColorMatrix")
            .replace("StaggeredFermion", "ColorVector")
            .replace("StaggeredPropagator", "ColorMatrix")
            .replace("Fermion", "SpinColorVector")
            .replace("Propagator", "SpinColorMatrix")
        )

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        location = "numpy" if isinstance(value, numpy.ndarray) else self.backend
        self._data = arrayAsContiguous(value, location)

    @property
    def data_ptr(self) -> NDArray:
        return self.data.reshape(-1)

    @property
    def data_ptrs(self) -> NDArray:
        if self.L5 == 0:
            getLogger().critical(f"{self.__class__.__name__}.data_ptrs not implemented", NotImplementedError)
        else:
            return self.data.reshape(self.L5, -1)

    @property
    def data_void_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)

    @classmethod
    def _field(cls) -> str:
        group_name = cls._groupName()
        return group_name[group_name.index("Lattice") + len("Lattice") :]

    def _setField(self):
        field_shape, field_dtype = _field_shape_dtype(self._field(), self.latt_info.Ns, self.latt_info.Nc)
        self.field_shape = field_shape
        self.field_dtype = field_dtype
        self.lattice_shape = self._latticeShape()
        self.shape: Tuple[int, ...] = (
            (*self.lattice_shape, *self.field_shape)
            if self.L5 == 0
            else (self.L5, *self.lattice_shape, *self.field_shape)
        )
        self.dtype = arrayDType(field_dtype, self.backend)

    def _initData(self, value: Union[NDArray, BackendType, None]):
        self._setField()
        if value is None:
            self.data = arrayZeros(self.shape, self.dtype, self.backend)
        elif isinstance(value, str):
            self.data = arrayZeros(self.shape, self.dtype, value)
        else:
            self.data = value.reshape(self.shape)

    def lexico(self, force_numpy: bool = True):
        if not isinstance(self, _FULL_FILED_TUPLE):
            getLogger().critical(f"{self.__class__.__name__}.lexico(force_numpy) not implemented", NotImplementedError)
        if force_numpy:
            return self.latt_info.lexico(self.getHost(), self.L5 > 0)
        else:
            return self.latt_info.lexico(self.data, self.L5 > 0, self.location)

    def _select_lattice(self, coord: Tuple[Union[int, list, tuple, slice], ...]):
        select = numpy.full(self.lattice_shape, True)
        shape = []
        subshape = []
        for i, index in enumerate(coord):
            X = self.latt_info.coordinate(i)
            GL = self.latt_info.global_size[i]
            g = self.latt_info.grid_coord[i]
            L = self.latt_info.size[i]
            subshape.append(0)
            if isinstance(index, int):
                shape.append(1)
                if g * L <= index % GL < (g + 1) * L:
                    subshape[-1] += 1
                select &= X == (index % GL)
            elif isinstance(index, (list, tuple)):
                shape.append(len(index))
                for i in index:
                    if g * L <= i % GL < (g + 1) * L:
                        subshape[-1] += 1
                subselect = X == (index[0] % GL)
                for i in range(1, len(index)):
                    subselect |= X == (index[i] % GL)
                select &= subselect
            elif isinstance(index, slice):
                start, stop, step = index.indices(GL)
                if start >= stop or step <= 0:
                    getLogger().critical(
                        f"{self.__class__.__name__}[...] only accepts a:b:c with a<b and c>0", ValueError
                    )
                shape.append((stop - start + step - 1) // step)
                for i in range(start, stop, step):
                    if g * L <= i < (g + 1) * L:
                        subshape[-1] += 1
                select &= (X >= start) & (X < stop)
                if step > 1:
                    select &= (X - start) % step == 0
            else:
                getLogger().critical(
                    f"{self.__class__.__name__}[...] only accepts "
                    f"int, list, tuple or slice as index, got {type(index)}",
                    ValueError,
                )
        select = numpy.where(select.reshape(-1))[0]
        return self.latt_info.sortSelect(select), shape, subshape

    def _select_field(self, field_coord, field_shape):
        if len(field_coord) == 0:
            return field_shape
        shape = []
        for index, L in zip(field_coord, field_shape):
            if isinstance(index, int):
                pass
            elif isinstance(index, (list, tuple)):
                shape.append(len(index))
            elif isinstance(index, slice):
                start, stop, step = index.indices(L)
                if start >= stop or step <= 0:
                    getLogger().critical(
                        f"{self.__class__.__name__}[...] only accepts a:b:c with a<b and c>0", ValueError
                    )
                shape.append((stop - start + step - 1) // step)
            else:
                getLogger().critical(
                    f"{self.__class__.__name__}[...] only accepts "
                    f"int, list, tuple or slice as index, got {type(index)}",
                    ValueError,
                )
        return shape

    def __setitem__(self, key: Tuple[Union[int, list, tuple, slice], ...], value: Union[int, float, complex, NDArray]):
        Nd = self.latt_info.Nd
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == Nd or len(key) == Nd + len(self.field_shape):
            select, shape, subshape = self._select_lattice(key[:Nd])
            field_shape = self._select_field(key[Nd:], self.field_shape)
            key_ = (select, *key[Nd:])
            if isinstance(value, (int, float, complex)):
                self.data.reshape(-1, *self.field_shape)[key_] = value
            elif isinstance(value, numpy.ndarray):
                if tuple(field_shape) == numpy.broadcast_shapes(field_shape, value.shape):
                    self.data.reshape(-1, *self.field_shape)[key_] = arrayDevice(value, self.location)
                else:
                    self.data.reshape(-1, *self.field_shape)[key_] = arrayDevice(value, self.location).reshape(
                        *subshape, *field_shape
                    )
            else:
                getLogger().critical(
                    f"{self.__class__.__name__}[...] only accepts "
                    f"int, float, complex or ndarray as value, got {type(value)}",
                    ValueError,
                )
        else:
            getLogger().critical(
                f"{self.__class__.__name__}[...] only accepts "
                f"{Nd} or {Nd + len(self.field_shape)} indices, got {len(key)}",
                ValueError,
            )

    def __getitem__(self, key: Tuple[Union[int, list, tuple, slice], ...]) -> NDArray:
        Nd = self.latt_info.Nd
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == Nd or len(key) == Nd + len(self.field_shape):
            select, shape, subshape = self._select_lattice(key[:Nd])
            field_shape = self._select_field(key[Nd:], self.field_shape)
            key_ = (select, *key[Nd:])
            return arrayHostCopy(self.data.reshape(-1, *self.field_shape)[key_], self.location).reshape(
                *subshape, *field_shape
            )
        else:
            getLogger().critical(
                f"{self.__class__.__name__}[...] only accepts "
                f"{Nd} or {Nd + len(self.field_shape)} indices, got {len(key)}",
                ValueError,
            )

    @classmethod
    def load(cls: Type[Self], filename: str) -> Self:
        if not issubclass(cls, _FULL_FILED_TUPLE):
            getLogger().critical(f"{cls.__name__}.load(filename) not implemented", NotImplementedError)
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        grid_size = getGridSize()
        Nd = len(grid_size)
        shape, dtype, offset = read_array_header(filename)
        latt_size = list(shape[:Nd][::-1])
        field_shape = list(shape[Nd:])
        sublatt_size = getSublatticeSize(latt_size, False)
        value = readMPIFile(filename, dtype, offset, [*sublatt_size[::-1], *field_shape], list(range(Nd - 1, -1, -1)))
        gbytes += value.nbytes / 1024**3
        Ns, Nc, dtype = _field_spin_color_dtype(cls._field(), field_shape, False)
        value = value.astype(dtype)
        Ns = 4 if Ns is None else Ns
        Nc = 3 if Nc is None else Nc
        if issubclass(cls, LexicoField):
            latt_info = LexicoInfo(latt_size, Ns=Ns, Nc=Nc)
            retval = cls(latt_info, value)
        elif issubclass(cls, FullField):
            latt_info = LatticeInfo(latt_size, Ns=Ns, Nc=Nc)
            retval = cls(latt_info, latt_info.evenodd(value, False))
        secs = perf_counter() - s
        getLogger().debug(f"Loaded {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")
        return retval

    def save(self, filename: str, *, use_fp32: bool = False):
        if not isinstance(self, _FULL_FILED_TUPLE):
            getLogger().critical(f"{self.__class__.__name__}.load(filename) not implemented", NotImplementedError)
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        if not filename.endswith(".npy"):
            filename += ".npy"
        field = self.lexico()
        _, _, dtype = _field_spin_color_dtype(self._field(), self.field_shape, use_fp32)
        if self.L5 == 0:
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
    def loadH5(cls: Type[Self], filename: str, label: Union[int, str], *, check: bool = True) -> Self:
        from .hdf5 import File

        if not issubclass(cls, _FULL_FILED_TUPLE):
            getLogger().critical(f"{cls.__name__}.load(filename, label, **kwargs) not implemented", NotImplementedError)
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        with File(filename, "r") as f:
            latt_size, Ns, Nc, value = f.load(cls._groupName(), label, check=check)
        Ns = 4 if Ns is None else Ns
        Nc = 3 if Nc is None else Nc
        if issubclass(cls, LexicoField):
            latt_info = LexicoInfo(latt_size, Ns=Ns, Nc=Nc)
            retval = cls(latt_info, value)
        elif issubclass(cls, FullField):
            latt_info = LatticeInfo(latt_size, Ns=Ns, Nc=Nc)
            retval = cls(latt_info, latt_info.evenodd(value, False))
        secs = perf_counter() - s
        getLogger().debug(f"Loaded {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")
        return retval

    def saveH5(
        self,
        filename: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        *,
        annotation: str = "",
        check: bool = True,
        use_fp32: bool = False,
    ):
        from .hdf5 import File

        if not isinstance(self, _FULL_FILED_TUPLE):
            getLogger().critical(
                f"{self.__class__.__name__}.save(filename, label, **kwargs) not implemented", NotImplementedError
            )
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        if not filename.endswith(".h5") and not filename.endswith(".hdf5"):
            filename += ".h5"
        with File(filename, "w") as f:
            f.save(self._groupName(), label, self.lexico(), annotation=annotation, check=check, use_fp32=use_fp32)
        secs = perf_counter() - s
        getLogger().debug(f"Saved {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")

    def appendH5(
        self,
        filename: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        *,
        annotation: str = "",
        check: bool = True,
        use_fp32: bool = False,
    ):
        from .hdf5 import File

        if not isinstance(self, _FULL_FILED_TUPLE):
            getLogger().critical(
                f"{self.__class__.__name__}.append(filename, label, **kwargs) not implemented", NotImplementedError
            )
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        with File(filename, "r+") as f:
            f.append(self._groupName(), label, self.lexico(), annotation=annotation, check=check, use_fp32=use_fp32)
        secs = perf_counter() - s
        getLogger().debug(f"Appended {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")

    def updateH5(
        self,
        filename: str,
        label: Union[int, str, Sequence[int], Sequence[str]],
        *,
        annotation: str = "",
        check: bool = True,
        use_fp32: bool = False,
    ):
        from .hdf5 import File

        if not isinstance(self, _FULL_FILED_TUPLE):
            getLogger().critical(
                f"{self.__class__.__name__}.update(filename, label, **kwargs) not implemented", NotImplementedError
            )
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        with File(filename, "r+") as f:
            f.update(self._groupName(), label, self.lexico(), annotation=annotation, check=check, use_fp32=use_fp32)
        secs = perf_counter() - s
        getLogger().debug(f"Updated {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")

    def copy(self):
        if self.L5 == 0:
            return self.__class__(self.latt_info, arrayCopy(self.data, self.location))
        else:
            return self.__class__(self.latt_info, self.L5, arrayCopy(self.data, self.location))

    def toDevice(self):
        self.data = arrayDevice(self.data, self.backend)

    def toHost(self):
        self.data = arrayHost(self.data, self.location)

    def getHost(self):
        return arrayHostCopy(self.data, self.location)

    def norm2(self, all_reduce=True) -> float:
        norm2 = arrayLinalgNorm(self.data, self.location) ** 2
        if all_reduce:
            return getMPIComm().allreduce(norm2)
        else:
            return norm2

    def __add__(self: Self, rhs: Self) -> Self:
        if not self.__class__ == rhs.__class__:
            return NotImplemented
        assert self.location == rhs.location
        if self.L5 == 0:
            return self.__class__(self.latt_info, self.data + rhs.data)
        else:
            return self.__class__(self.latt_info, self.L5, self.data + rhs.data)

    def __sub__(self: Self, rhs: Self) -> Self:
        if not self.__class__ == rhs.__class__:
            return NotImplemented
        assert self.location == rhs.location
        if self.L5 == 0:
            return self.__class__(self.latt_info, self.data - rhs.data)
        else:
            return self.__class__(self.latt_info, self.L5, self.data - rhs.data)

    def __mul__(self, rhs):
        if self.L5 == 0:
            return self.__class__(self.latt_info, self.data * rhs)
        else:
            return self.__class__(self.latt_info, self.L5, self.data * rhs)

    def __rmul__(self, lhs):
        if self.L5 == 0:
            return self.__class__(self.latt_info, lhs * self.data)
        else:
            return self.__class__(self.latt_info, self.L5, lhs * self.data)

    def __truediv__(self, rhs):
        if self.L5 == 0:
            return self.__class__(self.latt_info, self.data / rhs)
        else:
            return self.__class__(self.latt_info, self.L5, self.data / rhs)

    def __neg__(self):
        if self.L5 == 0:
            return self.__class__(self.latt_info, -self.data)
        else:
            return self.__class__(self.latt_info, self.L5, -self.data)

    def __iadd__(self: Self, rhs: Self) -> Self:
        if not self.__class__ == rhs.__class__:
            return NotImplemented
        assert self.location == rhs.location
        self._data += rhs.data
        return self

    def __isub__(self: Self, rhs: Self) -> Self:
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


class LexicoField(BaseField):
    latt_info: LexicoInfo

    def __init__(self, latt_info: LexicoInfo, value: Any = None, init_data: bool = True) -> None:
        super().__init__(latt_info)
        if init_data:
            self._initData(value)

    def _latticeShape(self):
        return [L for L in self.latt_info.size[::-1]]

    def shift(self, n: int, mu: int):
        assert 0 <= mu < 2 * self.latt_info.Nd
        Nd = self.latt_info.Nd
        direction = 1 if mu < Nd else -1
        direction *= 1 if n >= 0 else -1
        mu = mu % Nd
        n = abs(n)
        location = self.location
        left_slice = [slice(None, None) for _ in range(Nd)]
        right_slice = [slice(None, None) for _ in range(Nd)]
        left = arrayCopy(self.data, location)
        right = arrayCopy(self.data, location) if n > 1 else self.data
        rank = getMPIRank()
        coords = getGridCoord()
        g, G = coords[mu], getGridSize()[mu]
        coords[mu] = (g - direction) % G
        dest = getRankFromCoord(coords)
        coords[mu] = (g + direction) % G
        source = getRankFromCoord(coords)

        if n == 0:
            left, right = right, left
        while n > 0:
            left_slice[mu] = slice(-1, None) if direction == 1 else slice(None, 1)
            right_slice[mu] = slice(None, 1) if direction == 1 else slice(-1, None)
            sendbuf = right[tuple(right_slice[::-1])]
            if rank == source and rank == dest:
                pass
            else:
                sendbuf_host = arrayHostCopy(sendbuf, location)
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
                recvbuf = arrayDevice(recvbuf_host, location)
            left[tuple(left_slice[::-1])] = recvbuf

            n -= 1
            left, right = right, left

        assert isinstance(self.latt_info, LexicoInfo)
        return self.__class__(self.latt_info, right)


class ParityField(BaseField):
    latt_info: LatticeInfo

    def __init__(self, latt_info: LatticeInfo, value: Any = None, init_data: bool = True) -> None:
        super().__init__(latt_info)
        if init_data:
            self._initData(value)

    def _latticeShape(self):
        return [L // 2 if d == self.latt_info.Nd - 1 else L for d, L in enumerate(self.latt_info.size[::-1])]


class FullField(BaseField, Generic[Field]):
    latt_info: LatticeInfo

    def __init__(self, latt_info: LatticeInfo, value: Any = None, init_data: bool = True) -> None:
        s = super(FullField, self)
        if hasattr(s, "__field_class__"):
            getattr(s, "__field_class__").__base__.__init__(self, latt_info, value, False)
            # s.__field_class__.__base__.__init__(self, latt_info, value, False)
        else:
            s.__init__(latt_info, value, False)
        if init_data:
            self._initData(value)

    def _latticeShape(self):
        return [2] + [L // 2 if d == self.latt_info.Nd - 1 else L for d, L in enumerate(self.latt_info.size[::-1])]

    @property
    def even(self) -> Field:
        if self.L5 == 0:
            return getattr(super(FullField, self), "__field_class__")(self.latt_info, self.data[0])
            # return super(FullField, self).__field_class__(self.latt_info, self.data[0])
        else:
            getLogger().critical(f"{self.__class__.__name__}.even not implemented", NotImplementedError)

    @even.setter
    def even(self, value: Field):
        if self.L5 == 0:
            self.data[0] = value.data
        else:
            getLogger().critical(f"{self.__class__.__name__}.even not implemented", NotImplementedError)

    @property
    def even_ptr(self) -> NDArray:
        if self.L5 == 0:
            return self.data.reshape(2, -1)[0]
        else:
            getLogger().critical(f"{self.__class__.__name__}.even_ptr not implemented", NotImplementedError)

    @property
    def even_ptrs(self) -> NDArray:
        if self.L5 == 0:
            getLogger().critical(f"{self.__class__.__name__}.even_ptrs not implemented", NotImplementedError)
        else:
            return self.data.reshape(self.L5, 2, -1)[:, 0]

    @property
    def odd(self) -> Field:
        if self.L5 == 0:
            return getattr(super(FullField, self), "__field_class__")(self.latt_info, self.data[1])
            # return super(FullField, self).__field_class__(self.latt_info, self.data[1])
        else:
            getLogger().critical(f"{self.__class__.__name__}.odd not implemented", NotImplementedError)

    @odd.setter
    def odd(self, value: Field):
        if self.L5 == 0:
            self.data[1] = value.data
        else:
            getLogger().critical(f"{self.__class__.__name__}.odd not implemented", NotImplementedError)

    @property
    def odd_ptr(self) -> NDArray:
        if self.L5 == 0:
            return self.data.reshape(2, -1)[1]
        else:
            getLogger().critical(f"{self.__class__.__name__}.odd_ptr not implemented", NotImplementedError)

    @property
    def odd_ptrs(self) -> NDArray:
        if self.L5 == 0:
            getLogger().critical(f"{self.__class__.__name__}.odd_ptrs not implemented", NotImplementedError)
        else:
            return self.data.reshape(self.L5, 2, -1)[:, 1]

    def shift(self, n: int, mu: int):
        assert 0 <= mu < 2 * self.latt_info.Nd
        Nd = self.latt_info.Nd
        direction = 1 if mu < Nd else -1
        direction *= 1 if n >= 0 else -1
        mu = mu % Nd
        n = abs(n)
        location = self.location
        left_slice = [slice(None, None) for _ in range(Nd)]
        right_slice = [slice(None, None) for _ in range(Nd)]
        left = arrayCopy(self.data, location)
        right = arrayCopy(self.data, location) if n > 1 else self.data
        rank = getMPIRank()
        coords = getGridCoord()
        g, G = coords[mu], getGridSize()[mu]
        coords[mu] = (g - direction) % G
        dest = getRankFromCoord(coords)
        coords[mu] = (g + direction) % G
        source = getRankFromCoord(coords)

        if n == 0:
            left, right = right, left
        while n > 0:
            if mu == 0 and n == 1:
                Lx = self.latt_info.size[0]
                Sx = prod(self.latt_info.size[1:])
                left_flat = left.reshape(2, Sx, Lx // 2, -1)
                right_flat = right.reshape(2, Sx, Lx // 2, -1)
                eo = numpy.sum(numpy.indices((2, *self.latt_info.size[1:][::-1])), axis=0).reshape(-1) % 2
                even = numpy.where(eo == 0)[0].reshape(2, -1) % Sx
                odd = numpy.where(eo == 1)[0].reshape(2, -1) % Sx
                if direction == 1:
                    sendbuf = arrayDevice([right_flat[0, even[0], 0], right_flat[1, even[1], 0]], location)
                    if rank == source and rank == dest:
                        pass
                    else:
                        sendbuf_host = arrayHostCopy(sendbuf, location)
                        request = getMPIComm().Isend(sendbuf_host, dest)

                    left_flat[1, even[1]] = right_flat[0, odd[0]]
                    left_flat[0, even[0]] = right_flat[1, odd[1]]
                    left_flat[1, odd[1], :-1] = right_flat[0, even[0], 1:]
                    left_flat[0, odd[0], :-1] = right_flat[1, even[1], 1:]

                    if rank == source and rank == dest:
                        recvbuf = sendbuf
                    else:
                        recvbuf_host = numpy.empty_like(sendbuf_host)
                        getMPIComm().Recv(recvbuf_host, source)
                        request.Wait()
                        recvbuf = arrayDevice(recvbuf_host, location)
                    left_flat[1, odd[1], -1] = recvbuf[0]
                    left_flat[0, odd[0], -1] = recvbuf[1]
                else:
                    sendbuf = arrayDevice([right_flat[0, odd[0], -1], right_flat[1, odd[1], -1]], location)
                    if rank == source and rank == dest:
                        pass
                    else:
                        sendbuf_host = arrayHostCopy(sendbuf, location)
                        request = getMPIComm().Isend(sendbuf_host, dest)

                    left_flat[1, odd[1]] = right_flat[0, even[0]]
                    left_flat[0, odd[0]] = right_flat[1, even[1]]
                    left_flat[1, even[1], 1:] = right_flat[0, odd[0], :-1]
                    left_flat[0, even[0], 1:] = right_flat[1, odd[1], :-1]

                    if rank == source and rank == dest:
                        recvbuf = sendbuf
                    else:
                        recvbuf_host = numpy.empty_like(sendbuf_host)
                        getMPIComm().Recv(recvbuf_host, source)
                        request.Wait()
                        recvbuf = arrayDevice(recvbuf_host, location)
                    left_flat[1, even[1], 0] = recvbuf[0]
                    left_flat[0, even[0], 0] = recvbuf[1]

                n -= 1
            else:
                left_slice[mu] = slice(-1, None) if direction == 1 else slice(None, 1)
                right_slice[mu] = slice(None, 1) if direction == 1 else slice(-1, None)
                sendbuf = right[(slice(None, None),) + tuple(right_slice[::-1])]
                if rank == source and rank == dest:
                    pass
                else:
                    sendbuf_host = arrayHostCopy(sendbuf, location)
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
                    recvbuf = arrayDevice(recvbuf_host, location)
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


class MultiField(BaseField, Generic[Field]):
    def __init__(self, latt_info: BaseInfo, L5: int, value: Any = None, init_data: bool = True) -> None:
        s = super(MultiField, self)
        if hasattr(s, "__field_class__"):
            getattr(s, "__field_class__").__base__.__init__(self, latt_info, value, False)
            # s.__field_class__.__base__.__init__(self, latt_info, value, False)
        else:
            s.__init__(latt_info, value, False)
        assert L5 > 0
        self.L5 = L5
        if init_data:
            self._initData(value)

    @property
    def data(self) -> NDArray:
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
            self._data = arrayAsContiguous(value, location)

    @overload
    def __getitem__(self, key: int) -> Field: ...
    @overload
    def __getitem__(self: SelfMulti, key: Union[list, tuple, slice]) -> SelfMulti: ...

    def __getitem__(self: SelfMulti, key: Union[int, list, tuple, slice]) -> Union[Field, SelfMulti]:
        if isinstance(key, int):
            return getattr(super(MultiField, self), "__field_class__")(self.latt_info, self.data[key])
            # return super(MultiField, self).__field_class__(self.latt_info, self.data[key])
        elif isinstance(key, list):
            return self.__class__(self.latt_info, len(key), self.data[key])
        elif isinstance(key, tuple):
            return self.__class__(self.latt_info, len(key), self.data[list(key)])
        elif isinstance(key, slice):
            return self.__class__(self.latt_info, len(range(*key.indices(self.L5))), self.data[key])

    def __setitem__(self, key: Union[int, list, tuple, slice], value: Field):
        self.data[key] = value.data

    def shift(self: SelfMulti, n: Sequence[int], mu: Sequence[int]) -> SelfMulti:
        ret = self.copy()
        for i in range(self.L5):
            ret[i] = self[i].shift(n[i], mu[i])
        return ret

    @classmethod
    def load(cls: Type[SelfMulti], filename: str) -> SelfMulti:
        if not issubclass(cls, _FULL_FILED_TUPLE):
            getLogger().critical(f"{cls.__name__}.load(filename) not implemented", NotImplementedError)
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        grid_size = getGridSize()
        Nd = len(grid_size)
        shape, dtype, offset = read_array_header(filename)
        L5 = shape[0]
        latt_size = list(shape[1 : Nd + 1][::-1])
        field_shape = list(shape[Nd + 1 :])
        sublatt_size = getSublatticeSize(latt_size, False)
        value = readMPIFile(filename, dtype, offset, [L5, *sublatt_size[::-1], *field_shape], list(range(Nd, 0, -1)))
        gbytes += value.nbytes / 1024**3
        Ns, Nc, dtype = _field_spin_color_dtype(cls._field(), field_shape, False)
        value = value.astype(dtype)
        Ns = 4 if Ns is None else Ns
        Nc = 3 if Nc is None else Nc
        if issubclass(cls, LexicoField):
            latt_info = LexicoInfo(latt_size, Ns=Ns, Nc=Nc)
            retval = cls(latt_info, L5, value)
        elif issubclass(cls, FullField):
            latt_info = LatticeInfo(latt_size, Ns=Ns, Nc=Nc)
            retval = cls(latt_info, L5, latt_info.evenodd(value, True))
        secs = perf_counter() - s
        getLogger().debug(f"Loaded {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")
        return retval

    @classmethod
    def loadH5(
        cls: Type[SelfMulti], filename: str, label: Union[Sequence[int], Sequence[str]], *, check: bool = True
    ) -> SelfMulti:
        from .hdf5 import File

        if not issubclass(cls, _FULL_FILED_TUPLE):
            getLogger().critical(f"{cls.__name__}.load(filename, label, **kwargs) not implemented", NotImplementedError)
        s = perf_counter()
        gbytes = 0
        filename = path.expanduser(path.expandvars(filename))
        with File(filename, "r") as f:
            latt_size, Ns, Nc, value = f.load(cls._groupName(), label, check=check)
        Ns = 4 if Ns is None else Ns
        Nc = 3 if Nc is None else Nc
        if issubclass(cls, LexicoField):
            latt_info = LexicoInfo(latt_size, Ns=Ns, Nc=Nc)
            retval = cls(latt_info, len(label), value)
        elif issubclass(cls, FullField):
            latt_info = LatticeInfo(latt_size, Ns=Ns, Nc=Nc)
            retval = cls(latt_info, len(label), latt_info.evenodd(value, True))
        secs = perf_counter() - s
        getLogger().debug(f"Loaded {filename} in {secs:.3f} secs, {gbytes / secs:.3f} GB/s")
        return retval


_FULL_FILED_TUPLE = (LexicoField, FullField)


class LatticeInt(FullField, ParityField):
    @property
    def __field_class__(self):
        return LatticeInt


class MultiLatticeInt(MultiField[LatticeInt], LatticeInt):
    pass


class LatticeReal(FullField, ParityField):
    @property
    def __field_class__(self):
        return LatticeReal


class MultiLatticeReal(MultiField[LatticeReal], LatticeReal):
    pass


class LatticeComplex(FullField, ParityField):
    @property
    def __field_class__(self):
        return LatticeComplex


class MultiLatticeComplex(MultiField[LatticeComplex], LatticeComplex):
    pass


class LatticeLink(FullField, ParityField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info, value)
        if value is None:
            self.data[:] = arrayIdentity(latt_info.Nc, self.dtype, self.location)

    @property
    def __field_class__(self):
        return LatticeLink


class LatticeRotation(MultiField[LatticeLink], LatticeLink):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info, 1, value)
        if value is None:
            self.data[:] = arrayIdentity(latt_info.Nc, self.dtype, self.location)


class LatticeGauge(MultiField[LatticeLink], LatticeLink):
    def __init__(self, latt_info: LatticeInfo, L5: int, value=None) -> None:
        super().__init__(latt_info, L5, value)
        if value is None:
            self.data[:] = arrayIdentity(latt_info.Nc, self.dtype, self.location)

    def setAntiPeriodicT(self):
        if self.latt_info.gt == self.latt_info.Gt - 1:
            self.data[-1, :, -1] *= -1

    def setAnisotropy(self, anisotropy: float):
        self.data[:-1] /= anisotropy


class LatticeMom(MultiField, FullField, ParityField):
    pass


class LatticeClover(FullField, ParityField):
    pass


class HalfLatticeFermion(ParityField):
    @property
    def __field_class__(self):
        return HalfLatticeFermion


class MultiHalfLatticeFermion(MultiField[HalfLatticeFermion], HalfLatticeFermion):
    pass


class LatticeFermion(FullField[HalfLatticeFermion], HalfLatticeFermion):
    @property
    def __field_class__(self):
        return LatticeFermion


class MultiLatticeFermion(MultiField[LatticeFermion], LatticeFermion):
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


class MultiHalfLatticeStaggeredFermion(MultiField[HalfLatticeStaggeredFermion], HalfLatticeStaggeredFermion):
    pass


class LatticeStaggeredFermion(FullField[HalfLatticeStaggeredFermion], HalfLatticeStaggeredFermion):
    @property
    def __field_class__(self):
        return LatticeStaggeredFermion


class MultiLatticeStaggeredFermion(MultiField[LatticeStaggeredFermion], LatticeStaggeredFermion):
    pass


class LatticeStaggeredPropagator(FullField, ParityField):
    def setFermion(self, fermion: LatticeStaggeredFermion, color: int):
        self.data[:, :, :, :, :, :, color] = fermion.data

    def getFermion(self, color: int):
        return LatticeStaggeredFermion(self.latt_info, self.data[:, :, :, :, :, :, color])
