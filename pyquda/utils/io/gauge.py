import io
from os import path
import struct
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import numpy

from ...field import Ns, Nc, Nd, LatticeInfo, LatticeGauge, cb2

_precision_map = {"D": 8, "S": 4}


def gatherGaugeRaw(gauge_send: numpy.ndarray, latt_info: LatticeInfo):
    from ... import getMPIComm, getCoordFromRank

    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lt, Lz, Ly, Lx = latt_info.size
    dtype = gauge_send.dtype

    if latt_info.mpi_rank == 0:
        gauge_recv = numpy.zeros((Gt * Gz * Gy * Gx, Nd, Lt, Lz, Ly, Lx, Nc, Nc), dtype)
        getMPIComm().Gatherv(gauge_send, gauge_recv)

        gauge_raw = numpy.zeros((Nd, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc, Nc), dtype)
        for rank in range(latt_info.mpi_size):
            gx, gy, gz, gt = getCoordFromRank(rank, [Gx, Gy, Gz, Gt])
            gauge_raw[
                :,
                gt * Lt : (gt + 1) * Lt,
                gz * Lz : (gz + 1) * Lz,
                gy * Ly : (gy + 1) * Ly,
                gx * Lx : (gx + 1) * Lx,
            ] = gauge_recv[rank]
    else:
        gauge_recv = None
        getMPIComm().Gatherv(gauge_send, gauge_recv)

        gauge_raw = None

    return gauge_raw


def fromILDGBuffer(buffer: bytes, dtype: str, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    gauge_raw = (
        numpy.frombuffer(buffer, dtype)
        .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc, Nc)[
            gt * Lt : (gt + 1) * Lt,
            gz * Lz : (gz + 1) * Lz,
            gy * Ly : (gy + 1) * Ly,
            gx * Lx : (gx + 1) * Lx,
        ]
        .astype("<c16")
        .transpose(4, 0, 1, 2, 3, 5, 6)
    )

    return gauge_raw


def fromMILCBuffer(buffer: bytes, dtype: str, latt_info: LatticeInfo):
    """MILC and ILDG data have the exactly same layout."""
    return fromILDGBuffer(buffer, dtype, latt_info)


def fromKYUBuffer(buffer: bytes, dtype: str, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    gauge_raw = (
        numpy.frombuffer(buffer, dtype)
        .reshape(Nd, Nc, Nc, 2, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx)[
            :,
            :,
            :,
            :,
            gt * Lt : (gt + 1) * Lt,
            gz * Lz : (gz + 1) * Lz,
            gy * Ly : (gy + 1) * Ly,
            gx * Lx : (gx + 1) * Lx,
        ]
        .astype("<f8")
        .transpose(0, 4, 5, 6, 7, 2, 1, 3)
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2)
        .view("<c16")
    )

    return gauge_raw


def toKYUBuffer(gauge_lexico: numpy.ndarray, latt_info: LatticeInfo):
    Gx, Gy, Gz, Gt = latt_info.grid_size
    Lx, Ly, Lz, Lt = latt_info.size

    gauge_raw = gatherGaugeRaw(gauge_lexico, latt_info)
    if latt_info.mpi_rank == 0:
        buffer = (
            gauge_raw.view("<f8")
            .reshape(Nd, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc, Nc, 2)
            .transpose(0, 6, 5, 7, 1, 2, 3, 4)
            .astype(">f8")
            .tobytes()
        )
    else:
        buffer = None

    return buffer


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
    precision = _precision_map[scidac_private_record_xml.find("precision").text]
    assert int(scidac_private_record_xml.find("colors").text) == Nc
    assert (
        int(scidac_private_record_xml.find("spins").text) == Ns
        or int(scidac_private_record_xml.find("spins").text) == 1
    )
    assert int(scidac_private_record_xml.find("typesize").text) == Nc * Nc * 2 * precision
    assert int(scidac_private_record_xml.find("datacount").text) == Nd
    # latt_size = [
    #     int(ildg_format.find(f"{tag}lx").text),
    #     int(ildg_format.find(f"{tag}ly").text),
    #     int(ildg_format.find(f"{tag}lz").text),
    #     int(ildg_format.find(f"{tag}lt").text),
    # ]
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = map(int, scidac_private_file_xml.find("dims").text.split())
    latt_info = LatticeInfo(latt_size)
    gauge_raw = fromILDGBuffer(ildg_binary_data, f">c{2*precision}", latt_info)

    return LatticeGauge(latt_info, cb2(gauge_raw, [1, 2, 3, 4]))


def readILDGBin(filename: str, dtype: str, latt_size: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        ildg_binary_data = f.read()
    latt_info = LatticeInfo(latt_size)
    gauge_raw = fromILDGBuffer(ildg_binary_data, dtype, latt_info)

    return LatticeGauge(latt_info, cb2(gauge_raw, [1, 2, 3, 4]))


def readMILC(filename: str):
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
        # milc_binary_data = f.read(Lt * Lz * Ly * Lx * Nd * Nc * Nc * 2 * 4)
        milc_binary_data = f.read()
    # print(time_stamp, sum29, sum31)
    latt_info = LatticeInfo(latt_size)
    gauge_raw = fromMILCBuffer(milc_binary_data, f"{endian}c8", latt_info)

    return LatticeGauge(latt_info, cb2(gauge_raw, [1, 2, 3, 4]))


def readKYU(filename: str, latt_info: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        # kyu_binary_data = f.read(Nd * Nc * Nc * 2 * Lt * Lz * Ly * Lx * 8)
        kyu_binary_data = f.read()
    gauge_raw = fromKYUBuffer(kyu_binary_data, ">f8", latt_info)

    return LatticeGauge(latt_info, cb2(gauge_raw, [1, 2, 3, 4]))


def writeKYU(filename: str, gauge: LatticeGauge):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = gauge.latt_info
    kyu_binary_data = toKYUBuffer(gauge.lexico(), latt_info)
    if latt_info.mpi_rank == 0:
        with open(filename, "wb") as f:
            f.write(kyu_binary_data)
