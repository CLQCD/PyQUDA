import io
import struct
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import numpy

from .. import mpi
from ..field import Nc, cb2


def readStr(f: io.BufferedReader) -> str:
    length = struct.unpack(">i", f.read(4))[0]
    return f.read(length).decode("utf-8")


def readTuple(f: io.BufferedReader) -> Tuple[int]:
    length = struct.unpack(">i", f.read(4))[0]
    cnt = length // 4
    fmt = ">" + "i" * cnt
    return struct.unpack(fmt, f.read(4 * cnt))


def readPos(f: io.BufferedReader) -> int:
    return struct.unpack(">qq", f.read(16))[1]


def readVersion(f: io.BufferedReader) -> int:
    return struct.unpack(">i", f.read(4))[0]


def readTimeSlice(filename: str, Ne: int = None):
    with open(filename, "rb") as f:
        offsets: Dict[Tuple[int], int] = {}
        assert readStr(f) == "XXXXQDPLazyDiskMapObjFileXXXX"
        assert readVersion(f) == 1
        format = ET.ElementTree(ET.fromstring(readStr(f)))
        f.seek(readPos(f))
        num_records = struct.unpack(">I", f.read(4))[0]
        for _ in range(num_records):
            key = readTuple(f)
            val = readPos(f)
            offsets[key] = val
    precision = 32
    binary_dtype = f">c{2*precision//8}"
    ndarray_dtype = f"<c{2*precision//8}"
    latt_size = [int(x) for x in format.find("lattSize").text.split()]
    Lx, Ly, Lz, Lt = latt_size
    if Ne is None:
        Ne = int(format.find("num_vecs").text)

    Gx, Gy, Gz, Gt = mpi.grid
    gx, gy, gz, gt = mpi.coord
    latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
    Lx, Ly, Lz, Lt = latt_size

    eigen_raw = numpy.zeros((Ne, Lt, Lz, Ly, Lx, Nc), ndarray_dtype)
    for e in range(Ne):
        for t in range(Lt):
            eigen_raw[e, t] = (
                numpy.fromfile(
                    filename, binary_dtype, count=Gz * Lz * Gy * Ly * Gx * Lx * Nc, offset=offsets[(t + gt * Lt, e)]
                )
                .reshape(Gz * Lz, Gy * Ly, Gx * Lx, Nc)[
                    gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx, :
                ]
                .astype(ndarray_dtype)
            )

    return cb2(eigen_raw, [1, 2, 3, 4])
