import io
from os import path
import struct
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import numpy

from ...field import Ns, Nc, Nd, LatticeInfo, LatticePropagator, LatticeStaggeredPropagator, cb2

_precision_map = {"D": 8, "S": 4}


def fromSCIDACBuffer(buffer: bytes, dtype: str, latt_info: LatticeInfo, staggered: bool):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    if not staggered:
        propagator_raw = (
            numpy.frombuffer(buffer, dtype)
            .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Ns, Nc, Nc)[
                gt * Lt : (gt + 1) * Lt,
                gz * Lz : (gz + 1) * Lz,
                gy * Ly : (gy + 1) * Ly,
                gx * Lx : (gx + 1) * Lx,
            ]
            .astype("<c16")
        )
    else:
        propagator_raw = (
            numpy.frombuffer(buffer, dtype)
            .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc, Nc)[
                gt * Lt : (gt + 1) * Lt,
                gz * Lz : (gz + 1) * Lz,
                gy * Ly : (gy + 1) * Ly,
                gx * Lx : (gx + 1) * Lx,
            ]
            .astype("<c16")
        )

    return propagator_raw


def fromMultiSCIDACBuffer(buffer: bytes, dtype: str, latt_info: LatticeInfo, staggered: bool):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    if not staggered:
        from warnings import warn

        warn("WARNING: NOT sure about MILC QIO format for propagator!!!")
        propagator_raw = (
            numpy.frombuffer(buffer, dtype)
            .reshape(Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc)[
                :,
                :,
                gt * Lt : (gt + 1) * Lt,
                gz * Lz : (gz + 1) * Lz,
                gy * Ly : (gy + 1) * Ly,
                gx * Lx : (gx + 1) * Lx,
            ]
            .astype("<c16")
            .transpose(2, 3, 4, 5, 6, 0, 7, 1)
        )
    else:
        propagator_raw = (
            numpy.frombuffer(buffer, dtype)
            .reshape(Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc)[
                :,
                gt * Lt : (gt + 1) * Lt,
                gz * Lz : (gz + 1) * Lz,
                gy * Ly : (gy + 1) * Ly,
                gx * Lx : (gx + 1) * Lx,
            ]
            .astype("<c16")
            .transpose(1, 2, 3, 4, 5, 0)
        )

    return propagator_raw


def readChromaQIO(filename: str):
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

        f.seek(meta["scidac-private-file-xml"][0])
        scidac_private_file_xml = ET.ElementTree(
            ET.fromstring(f.read(meta["scidac-private-file-xml"][1]).strip(b"\x00").decode("utf-8"))
        )
        f.seek(meta["scidac-private-record-xml"][0])
        scidac_private_record_xml = ET.ElementTree(
            ET.fromstring(f.read(meta["scidac-private-record-xml"][1]).strip(b"\x00").decode("utf-8"))
        )
        f.seek(meta["scidac-binary-data"][0])
        scidac_binary_data = f.read(meta["scidac-binary-data"][1])
    precision = _precision_map[scidac_private_record_xml.find("precision").text]
    assert int(scidac_private_record_xml.find("colors").text) == Nc
    assert int(scidac_private_record_xml.find("spins").text) == Ns
    typesize = int(scidac_private_record_xml.find("typesize").text)
    if typesize == Nc * Nc * 2 * precision:
        staggered = True
    elif typesize == Ns * Ns * Nc * Nc * 2 * precision:
        staggered = False
    else:
        raise ValueError(f"Unknown typesize={typesize} in Chroma QIO propagator")
    assert int(scidac_private_record_xml.find("datacount").text) == 1
    dtype = f">c{2*precision}"
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = map(int, scidac_private_file_xml.find("dims").text.split())
    latt_info = LatticeInfo(latt_size)
    propagator_raw = fromSCIDACBuffer(scidac_binary_data, dtype, latt_info, staggered)

    if not staggered:
        return LatticePropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3]))
    else:
        return LatticeStaggeredPropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3]))


def readMILCQIO(filename: str):
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
        scidac_binary_data = b""
        for meta_scidac_binary_data in meta["scidac-binary-data"][1::2]:
            f.seek(meta_scidac_binary_data[0])
            scidac_binary_data += f.read(meta_scidac_binary_data[1])
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
    dtype = f">c{2*precision}"
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = map(int, scidac_private_file_xml.find("dims").text.split())
    latt_info = LatticeInfo(latt_size)
    propagator_raw = fromMultiSCIDACBuffer(scidac_binary_data, dtype, latt_info, staggered)

    if not staggered:
        return LatticePropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3]))
    else:
        return LatticeStaggeredPropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3]))
