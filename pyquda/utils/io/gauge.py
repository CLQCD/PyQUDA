import io
import struct
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import numpy

from ... import mpi
from ...field import Nc, Ns, Nd, cb2, LatticeGauge

precision_map = {"D": 8, "S": 4}


def fromILDGBuffer(buffer: bytes, dtype: str, latt_size: List[int]):
    Lx, Ly, Lz, Lt = latt_size
    Gx, Gy, Gz, Gt = mpi.grid
    gx, gy, gz, gt = mpi.coord
    latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
    Lx, Ly, Lz, Lt = latt_size

    gauge_raw = (
        numpy.frombuffer(buffer, dtype)
        .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc, Nc)[
            gt * Lt : (gt + 1) * Lt, gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx
        ]
        .astype("<c16")
        .transpose(4, 0, 1, 2, 3, 5, 6)
    )

    return LatticeGauge(latt_size, cb2(gauge_raw, [1, 2, 3, 4]))


def fromMILCBuffer(buffer: bytes, dtype: str, latt_size: List[int]):
    """MILC and ILDG data have the exactly same layout."""
    return fromILDGBuffer(buffer, dtype, latt_size)


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

        # f.seek(meta["ildg-format"][0])
        # ildg_format = ET.ElementTree(ET.fromstring(f.read(meta["ildg-format"][1]).strip(b"\x00").decode("utf-8")))
        f.seek(meta["scidac-private-file-xml"][0])
        scidac_private_file_xml = ET.ElementTree(
            ET.fromstring(f.read(meta["scidac-private-file-xml"][1]).strip(b"\x00").decode("utf-8"))
        )
        f.seek(meta["scidac-private-record-xml"][0])
        scidac_private_record_xml = ET.ElementTree(
            ET.fromstring(f.read(meta["scidac-private-record-xml"][1]).strip(b"\x00").decode("utf-8"))
        )
        f.seek(meta["ildg-binary-data"][0])
        ildg_binary_data = f.read(meta["ildg-binary-data"][1])
    # tag = re.match(r"\{.*\}", ildg_format.getroot().tag).group(0)
    # precision = int(ildg_format.find(f"{tag}precision").text)
    precision = precision_map[scidac_private_record_xml.find("precision").text]
    assert int(scidac_private_record_xml.find("colors").text) == Nc
    assert (
        int(scidac_private_record_xml.find("spins").text) == Ns
        or int(scidac_private_record_xml.find("spins").text) == 1
    )
    assert int(scidac_private_record_xml.find("typesize").text) == Nc * Nc * 2 * precision
    assert int(scidac_private_record_xml.find("datacount").text) == Nd
    dtype = f">c{2*precision}"
    # latt_size = [
    #     int(ildg_format.find(f"{tag}lx").text),
    #     int(ildg_format.find(f"{tag}ly").text),
    #     int(ildg_format.find(f"{tag}lz").text),
    #     int(ildg_format.find(f"{tag}lt").text),
    # ]
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = map(int, scidac_private_file_xml.find("dims").text.split())

    return fromILDGBuffer(ildg_binary_data, dtype, latt_size)


def readILDGBin(filename: str, dtype: str, latt_size: List[int]):
    with open(filename, "rb") as f:
        ildg_binary_data = f.read()

    return fromILDGBuffer(ildg_binary_data, dtype, latt_size)


def readMILC(filename: str):
    with open(filename, "rb") as f:
        magic = f.read(4)
        assert struct.unpack("<i", magic)[0] == 20103
        latt_size = struct.unpack("<iiii", f.read(16))
        time_stamp = f.read(64).decode()
        assert struct.unpack("<i", f.read(4))[0] == 0
        sum29, sum31 = struct.unpack("<II", f.read(8))
        # milc_binary_data = f.read(Lt * Lz * Ly * Lx * Nd * Nc * Nc * 2 * 4)
        milc_binary_data = f.read()

    return fromMILCBuffer(milc_binary_data, "<c8", latt_size)
