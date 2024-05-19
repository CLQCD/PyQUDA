import io
from os import path
import struct
from typing import Dict, Tuple
import zlib

import numpy as np

Nd, Ns, Nc = 4, 4, 3


def readQIOGauge(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        meta: Dict[str, Tuple[int]] = {}
        buffer = f.read(8)
        while buffer != b"" and buffer != b"\x0A":
            assert buffer.startswith(b"\x45\x67\x89\xAB\x00\x01")
            length = (struct.unpack(">Q", f.read(8))[0] + 7) // 8 * 8
            name = f.read(128).strip(b"\x00").decode("utf-8")
            meta[name] = (f.tell(), length)
            f.seek(length, io.SEEK_CUR)
            buffer = f.read(8)
        for key, value in meta.items():
            print(key)
            f.seek(value[0])
            if "binary" not in key and key not in ["scidac-file-xml", "scidac-record-xml"]:
                print(f.read(value[1]).strip(b"\x00").decode("utf-8"))
    return meta["ildg-binary-data"][0], ">c16", meta["ildg-binary-data"][1] // 16


def readQIOPropagator(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        meta: Dict[str, Tuple[int]] = {}
        buffer = f.read(8)
        while buffer != b"" and buffer != b"\x0A":
            assert buffer.startswith(b"\x45\x67\x89\xAB\x00\x01")
            length = (struct.unpack(">Q", f.read(8))[0] + 7) // 8 * 8
            name = f.read(128).strip(b"\x00").decode("utf-8")
            meta[name] = (f.tell(), length)
            f.seek(length, io.SEEK_CUR)
            buffer = f.read(8)
        for key, value in meta.items():
            print(key)
            f.seek(value[0])
            if "binary" not in key and key not in ["scidac-file-xml", "scidac-record-xml"]:
                print(f.read(value[1]).strip(b"\x00").decode("utf-8"))
    return meta["ildg-binary-data"][0], ">c16", meta["ildg-binary-data"][1] // 16


def readGauge(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        magic = f.read(4)
        for endian in ["<", ">"]:
            if struct.unpack(f"{endian}i", magic)[0] == 20103:
                break
        else:
            raise ValueError(f"Broken magic {magic} in MILC gauge")
        latt_size = struct.unpack(f"{endian}iiii", f.read(16))
        time_stamp = f.read(64).decode()
        assert struct.unpack(f"{endian}i", f.read(4))[0] == 0
        sum29, sum31 = struct.unpack(f"{endian}II", f.read(8))
        offset = f.tell()
    print(latt_size, time_stamp, sum29, sum31)
    return offset, f"{endian}c8", int(np.prod(latt_size)) * Nd * Nc * Nc


offset, dtype, count = readGauge("/public/ensemble/a09m310/l3296f211b630m0074m037m440e.4728")
buf = np.fromfile(
    "/public/ensemble/a09m310/l3296f211b630m0074m037m440e.4728",
    dtype=dtype,
    count=count,
    offset=offset,
)
work = buf.view("<u4")
rank = np.arange(96 * 32 * 32 * 32 * 4 * 3 * 3 * 8 // 4, dtype="<u4")
rank29 = rank % 29
rank31 = rank % 31
sum29 = np.bitwise_xor.reduce(np.bitwise_or(work << rank29, work >> (32 - rank29)))
sum31 = np.bitwise_xor.reduce(np.bitwise_or(work << rank31, work >> (32 - rank31)))
print(sum29, sum31)

offset, dtype, count = readQIOGauge("/public/ensemble/F32P30/beta6.41_mu-0.2295_ms-0.2050_L32x96_cfg_9000.lime")
buf = np.fromfile(
    "/public/ensemble/F32P30/beta6.41_mu-0.2295_ms-0.2050_L32x96_cfg_9000.lime",
    dtype=dtype,
    count=count,
    offset=offset,
)
buf = buf.reshape(96 * 32 * 32 * 32, 4 * 3 * 3)
work = np.empty(96 * 32 * 32 * 32, "<u4")
for i in range(96 * 32 * 32 * 32):
    work[i] = zlib.crc32(buf[i])
rank = np.arange(96 * 32 * 32 * 32, dtype="<u4")
rank29 = rank % 29
rank31 = rank % 31
sum29 = np.bitwise_xor.reduce(np.bitwise_or(work << rank29, work >> (32 - rank29)))
sum31 = np.bitwise_xor.reduce(np.bitwise_or(work << rank31, work >> (32 - rank31)))
print(hex(sum29)[2:], hex(sum31)[2:])
