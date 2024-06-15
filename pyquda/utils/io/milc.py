import io
from os import path
import struct
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import numpy

from ... import getSublatticeSize, readMPIFile

Nd, Ns, Nc = 4, 4, 3
_precision_map = {"D": 8, "F": 4, "S": 4}


def fromGaugeFile(filename: str, offset: int, dtype: str, sublatt_size: List[int]):
    Lx, Ly, Lz, Lt = sublatt_size

    gauge_raw = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0))
    gauge_raw = gauge_raw.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")

    return gauge_raw


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
        time_stamp = f.read(64).decode()  # noqa: F841
        assert struct.unpack(f"{endian}i", f.read(4))[0] == 0
        sum29, sum31 = struct.unpack(f"{endian}II", f.read(8))
        offset = f.tell()
    sublatt_size = getSublatticeSize(latt_size)
    gauge_raw = fromGaugeFile(filename, offset, f"{endian}c8", sublatt_size)
    return latt_size, gauge_raw


def fromMultiSCIDACPropagatorFile(
    filename: str, offset: List[int], dtype: str, sublatt_size: List[int], staggered: bool
):
    Lx, Ly, Lz, Lt = sublatt_size

    if not staggered:
        propagator_raw = numpy.empty((Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), dtype)
        for spin in range(Ns):
            for color in range(Nc):
                propagator_raw[spin, color] = readMPIFile(
                    filename, dtype, offset[spin * Nc + color], (Lt, Lz, Ly, Lx, Ns, Nc), (3, 2, 1, 0)
                )
        propagator_raw = propagator_raw.transpose(2, 3, 4, 5, 6, 0, 7, 1).astype("<c16")
    else:
        propagator_raw = numpy.empty((Nc, Lt, Lz, Ly, Lx, Nc), dtype)
        for color in range(Nc):
            propagator_raw[color] = readMPIFile(filename, dtype, offset[color], (Lt, Lz, Ly, Lx, Nc), (3, 2, 1, 0))
        propagator_raw = propagator_raw.transpose(1, 2, 3, 4, 5, 0).astype("<c16")

    return propagator_raw


def readQIOPropagator(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        meta: Dict[str, List[Tuple[int]]] = {}
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
    sublatt_size = getSublatticeSize(latt_size)
    propagator_raw = fromMultiSCIDACPropagatorFile(filename, offset, f">c{2*precision}", sublatt_size, staggered)
    return latt_size, staggered, propagator_raw
