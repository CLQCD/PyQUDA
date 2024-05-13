import io
from os import path
import struct
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import numpy

from ...field import Ns, Nc, Nd, LatticeInfo, LatticeGauge, LatticePropagator, LatticeStaggeredPropagator, cb2

_precision_map = {"D": 8, "F": 4, "S": 4}


def fromILDGGaugeBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from ... import openMPIFileRead, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileRead(filename)
    gauge_raw = numpy.empty((Lt, Lz, Ly, Lx, Nd, Nc, Nc), native_dtype)
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc, Nc),
        (Lt, Lz, Ly, Lx, Nd, Nc, Nc),
        (gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0, 0),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Read_all(gauge_raw)
    filetype.Free()
    fh.Close()

    return gauge_raw.transpose(4, 0, 1, 2, 3, 5, 6).view(dtype).astype("<c16")


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
        offset = meta["ildg-binary-data"][0]
    # tag = re.match(r"\{.*\}", ildg_format.getroot().tag).group(0)
    # precision = int(ildg_format.find(f"{tag}precision").text)
    # latt_size = [
    #     int(ildg_format.find(f"{tag}lx").text),
    #     int(ildg_format.find(f"{tag}ly").text),
    #     int(ildg_format.find(f"{tag}lz").text),
    #     int(ildg_format.find(f"{tag}lt").text),
    # ]
    precision = _precision_map[scidac_private_record_xml.find("precision").text]
    assert int(scidac_private_record_xml.find("colors").text) == Nc
    assert (
        int(scidac_private_record_xml.find("spins").text) == Ns
        or int(scidac_private_record_xml.find("spins").text) == 1
    )
    assert int(scidac_private_record_xml.find("typesize").text) == Nc * Nc * 2 * precision
    assert int(scidac_private_record_xml.find("datacount").text) == Nd
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = map(int, scidac_private_file_xml.find("dims").text.split())
    latt_info = LatticeInfo(latt_size)
    gauge_raw = fromILDGGaugeBuffer(filename, offset, f">c{2*precision}", latt_info)

    return LatticeGauge(latt_info, cb2(gauge_raw, [1, 2, 3, 4]))


def readILDGBinGauge(filename: str, dtype: str, latt_size: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = LatticeInfo(latt_size)
    gauge_raw = fromILDGGaugeBuffer(filename, 0, dtype, latt_info)

    return LatticeGauge(latt_info, cb2(gauge_raw, [1, 2, 3, 4]))


def fromSCIDACPropagatorBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo, staggered: bool):
    from ... import openMPIFileRead, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileRead(filename)
    if not staggered:
        propagator_raw = numpy.empty((Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), native_dtype)
        filetype = getMPIDatatype(native_dtype).Create_subarray(
            (Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Ns, Nc, Nc),
            (Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc),
            (gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0, 0, 0),
        )
    else:
        propagator_raw = numpy.empty((Lt, Lz, Ly, Lx, Nc, Nc), native_dtype)
        filetype = getMPIDatatype(native_dtype).Create_subarray(
            (Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc, Nc),
            (Lt, Lz, Ly, Lx, Nc, Nc),
            (gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
        )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Read_all(propagator_raw)
    filetype.Free()
    fh.Close()

    return propagator_raw.view(dtype).astype("<c16")


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

        f.seek(meta["scidac-private-file-xml"][0])
        scidac_private_file_xml = ET.ElementTree(
            ET.fromstring(f.read(meta["scidac-private-file-xml"][1]).strip(b"\x00").decode("utf-8"))
        )
        f.seek(meta["scidac-private-record-xml"][0])
        scidac_private_record_xml = ET.ElementTree(
            ET.fromstring(f.read(meta["scidac-private-record-xml"][1]).strip(b"\x00").decode("utf-8"))
        )
        offset = meta["scidac-binary-data"][0]
    precision = _precision_map[scidac_private_record_xml.find("precision").text]
    assert int(scidac_private_record_xml.find("colors").text) == Nc
    assert (
        int(scidac_private_record_xml.find("spins").text) == Ns
        or int(scidac_private_record_xml.find("spins").text) == 1
    )
    typesize = int(scidac_private_record_xml.find("typesize").text)
    if typesize == Nc * Nc * 2 * precision:
        staggered = True
    elif typesize == Ns * Ns * Nc * Nc * 2 * precision:
        staggered = False
    else:
        raise ValueError(f"Unknown typesize={typesize} in Chroma QIO propagator")
    assert int(scidac_private_record_xml.find("datacount").text) == 1
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = map(int, scidac_private_file_xml.find("dims").text.split())
    latt_info = LatticeInfo(latt_size)
    propagator_raw = fromSCIDACPropagatorBuffer(filename, offset, f">c{2*precision}", latt_info, staggered)

    if not staggered:
        return LatticePropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3]))
    else:
        return LatticeStaggeredPropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3]))
