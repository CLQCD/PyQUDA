import io
import struct
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import cupy as cp

from ..core import Nc


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


def readTimeSlice(filename: str):
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
    latt_size = [int(x) for x in format.find("lattSize").text.split()]
    Lx, Ly, Lz, Lt = latt_size
    num_vecs = int(format.find("num_vecs").text)
    eigen_raw = np.zeros((Lt, num_vecs, Lz, Ly, Lx, Nc), "<c16")
    eigen = np.zeros((Lt, num_vecs, 2, Lz, Ly, Lx // 2, Nc), "<c16")

    for t in range(Lt):
        for e in range(num_vecs):
            eigen_raw[t, e] = np.fromfile(
                filename,
                binary_dtype,
                count=Lz * Ly * Lx * Nc,
                offset=offsets[(t, e)],
            ).astype("<c16").reshape(Lz, Ly, Lx, Nc)

    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    eigen[t, :, 0, z, y, :, :] = eigen_raw[t, :, z, y, 0::2, :]
                    eigen[t, :, 1, z, y, :, :] = eigen_raw[t, :, z, y, 1::2, :]
                else:
                    eigen[t, :, 0, z, y, :, :] = eigen_raw[t, :, z, y, 1::2, :]
                    eigen[t, :, 1, z, y, :, :] = eigen_raw[t, :, z, y, 0::2, :]

    return eigen
