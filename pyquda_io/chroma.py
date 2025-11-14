from os import path
from typing import List

from pyquda_comm import getSublatticeSize, readMPIFile
from .io_utils import checksumSciDAC, gaugeReunitarize

Nd, Ns, Nc = 4, 4, 3
_precision_map = {"D": 8, "F": 4, "S": 4}


def readQIOGauge(filename: str, checksum: bool = True, reunitarize_sigma: float = 5e-7):
    from .lime import Lime

    filename = path.expanduser(path.expandvars(filename))
    lime = Lime(filename)
    scidac_private_file_xml = lime.loadXML("scidac-private-file-xml")
    scidac_private_record_xml = lime.loadXML("scidac-private-record-xml")
    scidac_checksum_xml = lime.loadXML("scidac-checksum")
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
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype = f">c{2 * precision}"

    gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0))
    if checksum:
        assert checksumSciDAC(latt_size, gauge.reshape(Lt * Lz * Ly * Lx, Nd * Nc * Nc)) == (
            int(scidac_checksum_xml.find("suma").text, 16),
            int(scidac_checksum_xml.find("sumb").text, 16),
        ), f"Bad checksum for {filename}"
    gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
    if precision == 4:
        gauge = gaugeReunitarize(gauge, reunitarize_sigma)  # 5e-7: Nc * 2**0.5 * 1.1920929e-07
    return latt_size, gauge


def readQIOPropagator(filename: str, checksum: bool = True):
    from .lime import Lime

    filename = path.expanduser(path.expandvars(filename))
    lime = Lime(filename)
    scidac_private_file_xml = lime.loadXML("scidac-private-file-xml")
    scidac_private_record_xml = lime.loadXML("scidac-private-record-xml")
    scidac_checksum_xml = lime.loadXML("scidac-checksum")
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
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype = f">c{2 * precision}"

    if not staggered:
        propagator = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc), (3, 2, 1, 0))
        if checksum:
            assert checksumSciDAC(latt_size, propagator.reshape(Lt * Lz * Ly * Lx, Ns * Ns * Nc * Nc)) == (
                int(scidac_checksum_xml.find("suma").text, 16),
                int(scidac_checksum_xml.find("sumb").text, 16),
            ), f"Bad checksum for {filename}"
        propagator = propagator.astype("<c16")
    else:
        propagator = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nc, Nc), (3, 2, 1, 0))
        if checksum:
            assert checksumSciDAC(latt_size, propagator.reshape(Lt * Lz * Ly * Lx, Nc * Nc)) == (
                int(scidac_checksum_xml.find("suma").text, 16),
                int(scidac_checksum_xml.find("sumb").text, 16),
            ), f"Bad checksum for {filename}"
        propagator = propagator.astype("<c16")
    return latt_size, staggered, propagator
