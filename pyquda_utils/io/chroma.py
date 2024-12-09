from os import path
from typing import List
from xml.etree import ElementTree as ET

from mpi4py import MPI

from .mpi_file import getSublatticeSize, getGridCoord, readMPIFile

Nd, Ns, Nc = 4, 4, 3
_precision_map = {"D": 8, "F": 4, "S": 4}


def checksum_qio(latt_size: List[int], grid_size: List[int], data):
    import zlib
    import numpy

    gx, gy, gz, gt = getGridCoord(grid_size)
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    gLx, gLy, gLz, gLt = gx * Lx, gy * Ly, gz * Lz, gt * Lt
    GLx, GLy, GLz, GLt = latt_size
    work = numpy.empty((Lt * Lz * Ly * Lx), "<u4")
    for i in range(Lt * Lz * Ly * Lx):
        work[i] = zlib.crc32(data[i])
    rank = (
        numpy.arange(GLt * GLz * GLy * GLx, dtype="<u8")
        .reshape(GLt, GLz, GLy, GLx)[gLt : gLt + Lt, gLz : gLz + Lz, gLy : gLy + Ly, gLx : gLx + Lx]
        .reshape(-1)
    )
    rank29 = (rank % 29).astype("<u4")
    rank31 = (rank % 31).astype("<u4")
    sum29 = MPI.COMM_WORLD.allreduce(numpy.bitwise_xor.reduce(work << rank29 | work >> (32 - rank29)), MPI.BXOR)
    sum31 = MPI.COMM_WORLD.allreduce(numpy.bitwise_xor.reduce(work << rank31 | work >> (32 - rank31)), MPI.BXOR)
    return sum29, sum31


def readQIOGauge(filename: str, grid_size: List[int], checksum: bool = True):
    from .lime import Lime

    lime = Lime(filename)
    scidac_private_file_xml = ET.ElementTree(
        ET.fromstring(lime.read("scidac-private-file-xml").strip(b"\x00").decode("utf-8"))
    )
    scidac_private_record_xml = ET.ElementTree(
        ET.fromstring(lime.read("scidac-private-record-xml").strip(b"\x00").decode("utf-8"))
    )
    scidac_checksum_xml = ET.ElementTree(ET.fromstring(lime.read("scidac-checksum").strip(b"\x00").decode("utf-8")))
    offset = lime.record("ildg-binary-data").offset

    precision = _precision_map[scidac_private_record_xml.find("precision").text]
    assert int(scidac_private_record_xml.find("colors").text) == Nc
    assert (
        int(scidac_private_record_xml.find("spins").text) == Ns
        or int(scidac_private_record_xml.find("spins").text) == 1
    )
    assert int(scidac_private_record_xml.find("typesize").text) == Nc * Nc * 2 * precision
    assert int(scidac_private_record_xml.find("datacount").text) == Nd
    assert int(scidac_private_file_xml.find("spacetime").text) == Nd
    latt_size = [int(L) for L in scidac_private_file_xml.find("dims").text.split()]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    dtype = f">c{2 * precision}"

    gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0), grid_size)
    if checksum:
        assert checksum_qio(latt_size, grid_size, gauge.reshape(Lt * Lz * Ly * Lx, Nd * Nc * Nc)) == (
            int(scidac_checksum_xml.find("suma").text, 16),
            int(scidac_checksum_xml.find("sumb").text, 16),
        ), f"Bad checksum for {filename}"
    gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
    return latt_size, gauge


def readILDGBinGauge(filename: str, dtype: str, latt_size: List[int], grid_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    offset = 0

    gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0), grid_size)
    gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
    return gauge


def readQIOPropagator(filename: str, grid_size: List[int], checksum: bool = True):
    from .lime import Lime

    lime = Lime(filename)
    scidac_private_file_xml = ET.ElementTree(
        ET.fromstring(lime.read("scidac-private-file-xml").strip(b"\x00").decode("utf-8"))
    )
    scidac_private_record_xml = ET.ElementTree(
        ET.fromstring(lime.read("scidac-private-record-xml").strip(b"\x00").decode("utf-8"))
    )
    scidac_checksum_xml = ET.ElementTree(ET.fromstring(lime.read("scidac-checksum").strip(b"\x00").decode("utf-8")))
    offset = lime.record("scidac-binary-data").offset

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
    latt_size = [int(L) for L in scidac_private_file_xml.find("dims").text.split()]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    dtype = f">c{2 * precision}"

    if not staggered:
        propagator = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), (3, 2, 1, 0), grid_size)
        if checksum:
            assert checksum_qio(latt_size, grid_size, propagator.reshape(Lt * Lz * Ly * Lx, Ns * Ns * Nc * Nc)) == (
                int(scidac_checksum_xml.find("suma").text, 16),
                int(scidac_checksum_xml.find("sumb").text, 16),
            ), f"Bad checksum for {filename}"
        propagator = propagator.astype("<c16")
    else:
        propagator = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nc, Nc), (3, 2, 1, 0), grid_size)
        if checksum:
            assert checksum_qio(latt_size, grid_size, propagator.reshape(Lt * Lz * Ly * Lx, Nc * Nc)) == (
                int(scidac_checksum_xml.find("suma").text, 16),
                int(scidac_checksum_xml.find("sumb").text, 16),
            ), f"Bad checksum for {filename}"
        propagator = propagator.astype("<c16")
    return latt_size, staggered, propagator
