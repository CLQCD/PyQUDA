from ctypes import Union, Structure, c_char, c_int, sizeof
from enum import IntEnum
import io
from os import path
from typing import Sequence

import numpy
from numpy.typing import NDArray


class _DimensionType(IntEnum):
    dim_other = 0
    dim_x = 1
    dim_y = 2
    dim_z = 3
    dim_t = 4
    dim_d = 5
    dim_c = 6
    dim_d2 = 7
    dim_c2 = 8
    dim_complex = 9
    dim_mass = 10
    dim_smear = 11
    dim_displacement = 12

    dim_s_01 = 13
    dim_s_02 = 14
    dim_s_03 = 15
    dim_s_11 = 16
    dim_s_12 = 17
    dim_s_13 = 18
    dim_d_01 = 19
    dim_d_02 = 20
    dim_d_03 = 21
    dim_d_11 = 22
    dim_d_12 = 23
    dim_d_13 = 24

    dim_conf = 25
    dim_operator = 26
    dim_momentum = 27
    dim_direction = 28
    dim_t2 = 29
    dim_mass2 = 30

    dim_column = 31
    dim_row = 32
    dim_temporary = 33
    dim_temporary2 = 34
    dim_temporary3 = 35
    dim_temporary4 = 36

    dim_errorbar = 37
    """0 means average, 1 means errorbar, ..."""

    dim_operator2 = 38

    dim_param = 39
    dim_fitleft = 40
    dim_fitright = 41

    dim_jackknife = 42
    dim_jackknife2 = 43
    dim_jackknife3 = 44
    dim_jackknife4 = 45

    dim_summary = 46
    """
    0 means average, 1 means standard deviation, 2 means minimal value, 3 means maximum value, 4 means standard error,
    5 means median, ...
    """

    dim_channel = 47
    dim_channel2 = 48

    dim_eigen = 49

    dim_d_row = 50
    """on matrix multiplication, row is contracted with the left operand, col is contracted with the right operand."""
    dim_d_col = 51
    dim_c_row = 52
    dim_c_col = 53

    dim_parity = 54
    """dimension for different parities. we use 1/-1 for +/- parities for baryons."""

    dim_noise = 55
    dim_evenodd = 56

    dim_disp_x = 57
    dim_disp_y = 58
    dim_disp_z = 59
    dim_disp_t = 60

    dim_t3 = 61
    dim_t4 = 62
    dim_t_source = 63
    dim_t_current = 64
    dim_t_sink = 65

    dim_nothing = 66
    """do not use this unless for unused data."""

    dim_bootstrap = 67

    # add new dimensions here and add a string name in xqcd_type_dim_desc[] in io_general.c
    # ...

    dim_last = 68


class _OneDim(Structure):
    # _pack_ = 1
    _fields_ = [("type", c_int), ("n_indices", c_int), ("indices", c_int * 1024)]

    type: int
    n_indices: int
    indices: Sequence[int]


class _Head(Structure):
    # _pack_ = 1
    _fields_ = [("n_dimensions", c_int), ("dimensions", _OneDim * 16)]

    n_dimensions: int
    dimensions: Sequence[_OneDim]

    @property
    def dimensions_type(self):
        return tuple([_DimensionType(self.dimensions[i].type)._name_[4:] for i in range(self.n_dimensions)])

    @property
    def dimensions_n_indices(self):
        return tuple([self.dimensions[i].n_indices for i in range(self.n_dimensions)])


class _FileType(Union):
    _fields_ = [("head", _Head), ("blank", c_char * 102400)]

    head: _Head
    blank: bytes

    @property
    def n_dimensions(self):
        return self.head.n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, value):
        self.head.n_dimensions = value

    @property
    def dimensions(self):
        return self.head.dimensions

    @property
    def dimensions_type(self):
        return self.head.dimensions_type

    @property
    def dimensions_n_indices(self):
        return self.head.dimensions_n_indices

    def __repr__(self):
        retval = ""
        for i in range(self.n_dimensions):
            retval += f"{_DimensionType(self.dimensions[i].type)._name_:18s}{self.dimensions[i].n_indices:<6d}( "
            for j in range(self.dimensions[i].n_indices):
                retval += f"{self.dimensions[i].indices[j]} "
            retval += ")\n"
        return retval


def read(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        head = _FileType.from_buffer_copy(f.read(sizeof(_FileType)))
        data = numpy.frombuffer(f.read(), "<f8").reshape(head.dimensions_n_indices)

    return head, data


def write(filename: str, head: _FileType, data: NDArray[numpy.float64]):
    assert head.dimensions_n_indices == data.shape
    assert data.dtype == numpy.float64

    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "wb") as f:
        f.write(bytes(head))
        f.write(data.tobytes())


class IOGeneral:
    def __init__(self, filename: str):
        self.filename = path.expanduser(path.expandvars(filename))
        with open(self.filename, "rb") as f:
            self.head = _FileType.from_buffer_copy(f.read(sizeof(_FileType)))
        self.data = None

    def read(self):
        with open(self.filename, "rb") as f:
            f.seek(sizeof(_FileType), io.SEEK_SET)
            self.data = numpy.frombuffer(f.read(), "<f8").reshape(self.head.dimensions_n_indices)

    def write(self):
        with open(self.filename, "wb") as f:
            f.write(bytes(self.head))
            f.write(self.data.reshape(self.head.dimensions_n_indices).tobytes())
