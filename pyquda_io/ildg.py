from os import path
from typing import List

from pyquda_comm import getSublatticeSize, readMPIFile
from .io_utils import checksumSciDAC, gaugeReunitarize

Nd, Ns, Nc = 4, 4, 3
ildg_xmlns = "http://www.lqcd.org/ildg"


def readGauge(filename: str, checksum: bool = True, reunitarize_sigma: float = 5e-7):
    from .lime import Lime

    filename = path.expanduser(path.expandvars(filename))
    lime = Lime(filename)
    ildg_format_xml = lime.loadXML("ildg-format")
    scidac_checksum_xml = lime.loadXML("scidac-checksum")
    offset = lime.record("ildg-binary-data").offset

    assert ildg_format_xml.find(f"{{{ildg_xmlns}}}field").text == "su3gauge"
    precision = int(ildg_format_xml.find(f"{{{ildg_xmlns}}}precision").text) // 8
    latt_size = [
        int(ildg_format_xml.find(f"{{{ildg_xmlns}}}lx").text),
        int(ildg_format_xml.find(f"{{{ildg_xmlns}}}ly").text),
        int(ildg_format_xml.find(f"{{{ildg_xmlns}}}lz").text),
        int(ildg_format_xml.find(f"{{{ildg_xmlns}}}lt").text),
    ]
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


def readBinGauge(filename: str, dtype: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    offset = 0

    gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0))
    gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
    return gauge
