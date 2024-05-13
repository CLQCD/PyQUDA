import io
from os import path
import struct
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import numpy

from ...field import Ns, Nc, Nd, LatticeInfo, LatticeGauge, LatticePropagator, LatticeStaggeredPropagator, cb2

_precision_map = {"D": 8, "F": 4, "S": 4}


def fromGaugeBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
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
    latt_info = LatticeInfo(latt_size)
    gauge_raw = fromGaugeBuffer(filename, offset, f"{endian}c8", latt_info)

    return LatticeGauge(latt_info, cb2(gauge_raw, [1, 2, 3, 4]))


def fromMultiSCIDACPropagatorBuffer(
    filename: str, offset: List[int], dtype: str, latt_info: LatticeInfo, staggered: bool
):
    from ... import openMPIFileRead, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileRead(filename)
    if not staggered:
        propagator_raw = numpy.empty((Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), native_dtype)
        filetype = getMPIDatatype(native_dtype).Create_subarray(
            (Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc),
            (Lt, Lz, Ly, Lx, Ns, Nc),
            (gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
        )
        filetype.Commit()
        for spin in range(Ns):
            for color in range(Nc):
                fh.Set_view(offset[spin * Nc + color], filetype=filetype)
                fh.Read_all(propagator_raw[spin, color])
        filetype.Free()
        propagator_raw = propagator_raw.transpose(2, 3, 4, 5, 6, 0, 7, 1).view(dtype).astype("<c16")
    else:
        propagator_raw = numpy.empty((Nc, Lt, Lz, Ly, Lx, Nc), native_dtype)
        filetype = getMPIDatatype(native_dtype).Create_subarray(
            (Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc),
            (Lt, Lz, Ly, Lx, Nc),
            (gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0),
        )
        filetype.Commit()
        for color in range(Nc):
            fh.Set_view(offset[color], filetype=filetype)
            fh.Read_all(propagator_raw[color])
        filetype.Free()
        propagator_raw = propagator_raw.transpose(1, 2, 3, 4, 5, 0).view(dtype).astype("<c16")
    fh.Close()

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
    latt_size = map(int, scidac_private_file_xml.find("dims").text.split())
    latt_info = LatticeInfo(latt_size)
    propagator_raw = fromMultiSCIDACPropagatorBuffer(filename, offset, f">c{2*precision}", latt_info, staggered)

    if not staggered:
        return LatticePropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3]))
    else:
        return LatticeStaggeredPropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3]))
