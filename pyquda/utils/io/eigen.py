import io
from os import path
import struct
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import numpy

from ...field import cb2

Nc = 3


def _readStr(f: io.BufferedReader) -> str:
    length = struct.unpack(">i", f.read(4))[0]
    return f.read(length).decode("utf-8")


def _readTuple(f: io.BufferedReader) -> Tuple[int]:
    length = struct.unpack(">i", f.read(4))[0]
    cnt = length // 4
    fmt = ">" + "i" * cnt
    return struct.unpack(fmt, f.read(4 * cnt))


def _readPos(f: io.BufferedReader) -> int:
    return struct.unpack(">qq", f.read(16))[1]


def _readVersion(f: io.BufferedReader) -> int:
    return struct.unpack(">i", f.read(4))[0]


def readTimeSlice(filename: str, Ne: int = None):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        offsets: Dict[Tuple[int], int] = {}
        assert _readStr(f) == "XXXXQDPLazyDiskMapObjFileXXXX"
        assert _readVersion(f) == 1
        format = ET.ElementTree(ET.fromstring(_readStr(f)))
        f.seek(_readPos(f))
        num_records = struct.unpack(">I", f.read(4))[0]
        for _ in range(num_records):
            key = _readTuple(f)
            val = _readPos(f)
            offsets[key] = val
    precision = 32
    binary_dtype = f">c{2*precision//8}"
    ndarray_dtype = f"<c{2*precision//8}"
    latt_size = [int(x) for x in format.find("lattSize").text.split()]
    if Ne is None:
        Ne = int(format.find("num_vecs").text)

    from ... import getGridSize, getGridCoord, getSublatticeSize

    Gx, Gy, Gz, Gt = getGridSize()
    gx, gy, gz, gt = getGridCoord()
    sublatt_size = getSublatticeSize(latt_size)
    Lx, Ly, Lz, Lt = sublatt_size

    eigen_raw = numpy.zeros((Ne, Lt, Lz, Ly, Lx, Nc), ndarray_dtype)
    for e in range(Ne):
        for t in range(Lt):
            eigen_raw[e, t] = (
                numpy.fromfile(
                    filename, binary_dtype, count=Gz * Lz * Gy * Ly * Gx * Lx * Nc, offset=offsets[(t + gt * Lt, e)]
                )
                .reshape(Gz * Lz, Gy * Ly, Gx * Lx, Nc)[
                    gz * Lz : (gz + 1) * Lz,
                    gy * Ly : (gy + 1) * Ly,
                    gx * Lx : (gx + 1) * Lx,
                ]
                .astype(ndarray_dtype)
            )

    return cb2(eigen_raw, [1, 2, 3, 4])
