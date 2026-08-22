from os import path
from typing import List

from .mpi_utils import getSublatticeSize, readMPIFile
from .io_utils import checksumSciDAC, gaugeReunitarize

Nd, Ns, Nc = 4, 4, 3
ildg_xmlns = {"": "http://www.lqcd.org/ildg"}


def readGauge(filename: str, checksum: bool = True, reunitarize_sigma: float = 1e-6):
    from .lime import Lime

    filename = path.expanduser(path.expandvars(filename))
    lime = Lime(filename)
    ildg_format_xml = lime.loadXML("ildg-format")
    scidac_checksum_xml = lime.loadXML("scidac-checksum")
    offset = lime.record("ildg-binary-data").offset

    assert ildg_format_xml.find("field", ildg_xmlns).text == "su3gauge"
    precision = int(ildg_format_xml.find("precision", ildg_xmlns).text) // 8
    latt_size = [
        int(ildg_format_xml.find("lx", ildg_xmlns).text),
        int(ildg_format_xml.find("ly", ildg_xmlns).text),
        int(ildg_format_xml.find("lz", ildg_xmlns).text),
        int(ildg_format_xml.find("lt", ildg_xmlns).text),
    ]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    dtype = f">c{2 * precision}"

    gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0))
    if checksum:
        assert (
            int(scidac_checksum_xml.find("suma").text, 16),
            int(scidac_checksum_xml.find("sumb").text, 16),
        ) == checksumSciDAC(latt_size, gauge.reshape(Lt * Lz * Ly * Lx, Nd * Nc * Nc))
    gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
    if precision == 4:
        gauge = gaugeReunitarize(gauge, reunitarize_sigma)
    return latt_size, gauge


def readBinGauge(filename: str, dtype: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    offset = 0

    gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0))
    gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
    return gauge
