from datetime import datetime
import io
from os import path
import struct
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import numpy

from pyquda import getSublatticeSize, getMPIComm, getMPISize, getMPIRank, getGridCoord, readMPIFile, writeMPIFile

Nd, Ns, Nc = 4, 4, 3
_precision_map = {"D": 8, "F": 4, "S": 4}


def checksum_milc(latt_size: List[int], data):
    import numpy
    from mpi4py import MPI

    gx, gy, gz, gt = getGridCoord()
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    gLx, gLy, gLz, gLt = gx * Lx, gy * Ly, gz * Lz, gt * Lt
    GLx, GLy, GLz, GLt = latt_size

    work = data.view("<u4")
    rank = (
        numpy.arange(getMPISize() * work.size, dtype="<u4")
        .reshape(GLt, GLz, GLy, GLx, -1)[gLt : gLt + Lt, gLz : gLz + Lz, gLy : gLy + Ly, gLx : gLx + Lx]
        .reshape(-1)
    )
    rank29 = rank % 29
    rank31 = rank % 31
    sum29 = getMPIComm().allreduce(numpy.bitwise_xor.reduce(work << rank29 | work >> (32 - rank29)).item(), MPI.BXOR)
    sum31 = getMPIComm().allreduce(numpy.bitwise_xor.reduce(work << rank31 | work >> (32 - rank31)).item(), MPI.BXOR)
    return sum29, sum31


def checksum_qio(latt_size: List[int], data):
    import zlib
    import numpy
    from mpi4py import MPI

    gx, gy, gz, gt = getGridCoord()
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    gLx, gLy, gLz, gLt = gx * Lx, gy * Ly, gz * Lz, gt * Lt
    GLx, GLy, GLz, GLt = latt_size
    work = numpy.empty((Lt * Lz * Ly * Lx), "<u4")
    for i in range(Lt * Lz * Ly * Lx):
        work[i] = zlib.crc32(data[i])
    rank = (
        numpy.arange(GLt * GLz * GLy * GLx, dtype="<u4")
        .reshape(GLt, GLz, GLy, GLx)[gLt : gLt + Lt, gLz : gLz + Lz, gLy : gLy + Ly, gLx : gLx + Lx]
        .reshape(-1)
    )
    rank29 = rank % 29
    rank31 = rank % 31
    sum29 = getMPIComm().allreduce(numpy.bitwise_xor.reduce(work << rank29 | work >> (32 - rank29)).item(), MPI.BXOR)
    sum31 = getMPIComm().allreduce(numpy.bitwise_xor.reduce(work << rank31 | work >> (32 - rank31)).item(), MPI.BXOR)
    return sum29, sum31


def readGauge(filename: str, checksum: bool = True):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        magic = f.read(4)
        for endian in ["<", ">"]:
            if struct.unpack(f"{endian}i", magic)[0] == 20103:
                break
        else:
            raise ValueError(f"Broken magic {magic} in MILC gauge")
        latt_size = struct.unpack(f"{endian}iiii", f.read(16))
        time_stamp = f.read(64).decode()  # noqa: F841
        assert struct.unpack(f"{endian}i", f.read(4))[0] == 0  # order
        sum29, sum31 = struct.unpack(f"{endian}II", f.read(8))
        offset = f.tell()
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype = f"{endian}c8"

    gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0))
    if checksum:
        assert checksum_milc(latt_size, gauge.astype("<c8").reshape(-1)) == (
            sum29,
            sum31,
        ), f"Bad checksum for {filename}"
    gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
    return latt_size, gauge


def writeGauge(filename: str, latt_size: List[int], gauge: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", None

    gauge = numpy.ascontiguousarray(gauge.transpose(1, 2, 3, 4, 0, 5, 6).astype(dtype))
    sum29, sum31 = checksum_milc(latt_size, gauge.reshape(-1))
    if getMPIRank() == 0:
        with open(filename, "wb") as f:
            f.write(struct.pack("<i", 20103))
            f.write(struct.pack("<iiii", *latt_size))
            f.write(datetime.now().strftime(R"%a %b %d %H:%M:%S %Y").encode().ljust(64, b"\x00"))
            f.write(struct.pack("<i", 0))  # order
            f.write(struct.pack("<II", sum29, sum31))
            offset = f.tell()
    offset = getMPIComm().bcast(offset)

    writeMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0), gauge)


def readQIOPropagator(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        meta: Dict[str, List[Tuple[int, int]]] = {}
        buffer = f.read(8)
        while buffer != b"" and buffer != b"\x0A":
            assert buffer.startswith(b"\x45\x67\x89\xAB\x00\x01")
            length = (struct.unpack(">Q", f.read(8))[0] + 7) // 8 * 8
            name = f.read(128).strip(b"\x00").decode("utf-8")
            if name not in meta:
                meta[name] = [(f.tell(), length)]
            else:
                meta[name].append((f.tell(), length))
            f.seek(length, io.SEEK_CUR)
            buffer = f.read(8)

        f.seek(meta["scidac-private-file-xml"][0][0])
        scidac_private_file_xml = ET.ElementTree(
            ET.fromstring(f.read(meta["scidac-private-file-xml"][0][1]).strip(b"\x00").decode("utf-8"))
        )
        f.seek(meta["scidac-private-record-xml"][1][0])
        scidac_private_record_xml = ET.ElementTree(
            ET.fromstring(f.read(meta["scidac-private-record-xml"][1][1]).strip(b"\x00").decode("utf-8"))
        )
        offset = []
        for meta_scidac_binary_data in meta["scidac-binary-data"][1::2]:
            offset.append(meta_scidac_binary_data[0])
    precision = _precision_map[scidac_private_record_xml.find("precision").text]
    assert int(scidac_private_record_xml.find("colors").text) == Nc
    if scidac_private_record_xml.find("spins") is not None:
        assert int(scidac_private_record_xml.find("spins").text) == Ns
    typesize = int(scidac_private_record_xml.find("typesize").text)
    if typesize == Nc * 2 * precision:
        staggered = True
    elif typesize == Ns * Nc * 2 * precision:
        staggered = False
    else:
        raise ValueError(f"Unknown typesize={typesize} in MILC QIO propagator")
    assert int(scidac_private_record_xml.find("datacount").text) == 1
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = [int(L) for L in scidac_private_file_xml.find("dims").text.split()]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype = f">c{2 * precision}"

    if not staggered:
        propagator = numpy.empty((Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), dtype)
        for spin in range(Ns):
            for color in range(Nc):
                propagator[spin, color] = readMPIFile(
                    filename, dtype, offset[spin * Nc + color], (Lt, Lz, Ly, Lx, Ns, Nc), (3, 2, 1, 0)
                )
        propagator = propagator.transpose(2, 3, 4, 5, 6, 0, 7, 1).astype("<c16")
    else:
        propagator = numpy.empty((Nc, Lt, Lz, Ly, Lx, Nc), dtype)
        for color in range(Nc):
            propagator[color] = readMPIFile(filename, dtype, offset[color], (Lt, Lz, Ly, Lx, Nc), (3, 2, 1, 0))
        propagator = propagator.transpose(1, 2, 3, 4, 5, 0).astype("<c16")
    return latt_size, staggered, propagator
