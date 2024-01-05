import io
import struct
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import numpy

from ...field import Ns, Nc, Nd, LatticeInfo, LatticePropagator, LatticeStaggeredPropagator, cb2

precision_map = {"D": 8, "S": 4}


def fromSCIDACBuffer(buffer: bytes, dtype: str, latt_info: LatticeInfo, staggered: bool):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    if not staggered:
        propagator_raw = (
            numpy.frombuffer(buffer, dtype)
            .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Ns, Nc, Nc)[
                gt * Lt : (gt + 1) * Lt, gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx
            ]
            .astype("<c16")
        )
        return LatticePropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3]))
    else:
        propagator_raw = (
            numpy.frombuffer(buffer, dtype)
            .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc, Nc)[
                gt * Lt : (gt + 1) * Lt, gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx
            ]
            .astype("<c16")
        )
        return LatticeStaggeredPropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3]))


def readQIO(filename: str):
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
    precision = precision_map[scidac_private_record_xml.find("precision").text]
    assert int(scidac_private_record_xml.find("colors").text) == Nc
    assert int(scidac_private_record_xml.find("spins").text) == Ns
    typesize = int(scidac_private_record_xml.find("typesize").text)
    if typesize == Nc * Nc * 2 * precision:
        staggered = True
    elif typesize == Ns * Ns * Nc * Nc * 2 * precision:
        staggered = False
    else:
        raise ValueError(f"Unknown typesize = {typesize} in QIO propagator")
    assert int(scidac_private_record_xml.find("datacount").text) == 1
    dtype = f">c{2*precision}"
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = map(int, scidac_private_file_xml.find("dims").text.split())
    latt_info = LatticeInfo(latt_size, 1, 1)

    return fromSCIDACBuffer(scidac_binary_data, dtype, latt_info, staggered)
