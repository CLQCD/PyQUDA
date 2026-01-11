from datetime import datetime
from os import path
import struct
from typing import List

import numpy

from pyquda_comm import getSublatticeSize, openReadHeader, openWriteHeader, readMPIFile, writeMPIFile
from .io_utils import checksumMILC, gaugeReunitarize

Nd, Ns, Nc = 4, 4, 3
_precision_map = {"D": 8, "F": 4}


def readGauge(filename: str, checksum: bool = True, reunitarize_sigma: float = 5e-7):
    filename = path.expanduser(path.expandvars(filename))
    with openReadHeader(filename) as f:
        if f.fp is not None:
            magic = f.fp.read(4)
            for endian in ["<", ">"]:
                if struct.unpack(f"{endian}i", magic)[0] == 20103:
                    break
            else:
                raise ValueError(f"Broken magic {magic} in MILC gauge")
            latt_size = list(struct.unpack(f"{endian}iiii", f.fp.read(16)))
            timestamp = f.fp.read(64).decode()  # noqa: F841
            assert struct.unpack(f"{endian}i", f.fp.read(4))[0] == 0  # order
            sum29, sum31 = struct.unpack(f"{endian}II", f.fp.read(8))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = f"{endian}c8", f.offset

    gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0))
    if checksum:
        assert (sum29, sum31) == checksumMILC(latt_size, gauge.astype("<c8").reshape(-1))
    gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
    gauge = gaugeReunitarize(gauge, reunitarize_sigma)  # 5e-7: Nc * 2**0.5 * 1.1920929e-07
    return latt_size, gauge


def writeGauge(filename: str, latt_size: List[int], gauge: numpy.ndarray):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype, offset = "<c8", None

    gauge = numpy.ascontiguousarray(gauge.transpose(1, 2, 3, 4, 0, 5, 6).astype(dtype))
    sum29, sum31 = checksumMILC(latt_size, gauge.reshape(-1))
    with openWriteHeader(filename) as f:
        if f.fp is not None:
            f.fp.write(struct.pack("<i", 20103))
            f.fp.write(struct.pack("<iiii", *latt_size))
            f.fp.write(datetime.now().strftime("%c").encode().ljust(64, b"\x00"))
            f.fp.write(struct.pack("<i", 0))  # order
            f.fp.write(struct.pack("<II", sum29, sum31))
    offset = f.offset

    writeMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0), gauge)


def readQIOPropagator(filename: str):
    from .lime import Lime

    filename = path.expanduser(path.expandvars(filename))
    lime = Lime(filename)
    scidac_private_file_xml = lime.loadXML("scidac-private-file-xml", 0)
    scidac_private_record_xml = lime.loadXML("scidac-private-record-xml", 1)
    offset = [record.offset for record in lime.records("scidac-binary-data")[1::2]]

    precision = _precision_map[scidac_private_record_xml.find("precision").text]
    assert int(scidac_private_record_xml.find("colors").text) == Nc
    assert int(scidac_private_record_xml.find("spins").text) == Ns
    assert int(scidac_private_record_xml.find("typesize").text) == Ns * Nc * 2 * precision
    assert int(scidac_private_record_xml.find("datacount").text) == 1
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = [int(L) for L in scidac_private_file_xml.find("dims").text.split()]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype = f">c{2 * precision}"

    propagator = numpy.empty((Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), dtype)
    for spin in range(Ns):
        for color in range(Nc):
            propagator[spin, color] = readMPIFile(
                filename, dtype, offset[spin * Nc + color], (Lt, Lz, Ly, Lx, Ns, Nc), (3, 2, 1, 0)
            )
    propagator = propagator.transpose(2, 3, 4, 5, 6, 0, 7, 1).astype("<c16")
    return latt_size, propagator


def readQIOStaggeredPropagator(filename: str):
    from .lime import Lime

    filename = path.expanduser(path.expandvars(filename))
    lime = Lime(filename)
    scidac_private_file_xml = lime.loadXML("scidac-private-file-xml", 0)
    scidac_private_record_xml = lime.loadXML("scidac-private-record-xml", 1)
    offset = [record.offset for record in lime.records("scidac-binary-data")[1::2]]

    precision = _precision_map[scidac_private_record_xml.find("precision").text]
    assert int(scidac_private_record_xml.find("colors").text) == Nc
    assert int(scidac_private_record_xml.find("typesize").text) == Nc * 2 * precision
    assert int(scidac_private_record_xml.find("datacount").text) == 1
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = [int(L) for L in scidac_private_file_xml.find("dims").text.split()]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype = f">c{2 * precision}"

    propagator = numpy.empty((Nc, Lt, Lz, Ly, Lx, Nc), dtype)
    for color in range(Nc):
        propagator[color] = readMPIFile(filename, dtype, offset[color], (Lt, Lz, Ly, Lx, Nc), (3, 2, 1, 0))
    propagator = propagator.transpose(1, 2, 3, 4, 5, 0).astype("<c16")
    return latt_size, propagator
