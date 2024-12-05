from os import path
from typing import Dict

import numpy

from pyquda import getSublatticeSize, readMPIFile, writeMPIFile, getMPIComm, getMPIRank

Nd, Ns, Nc = 4, 4, 3


def checksum_nersc(data: numpy.ndarray) -> int:
    from mpi4py import MPI

    return getMPIComm().allreduce(numpy.sum(data.view("<u4"), dtype="<u4"), MPI.SUM)


def readGauge(filename: str, link_trace: bool = True, checksum: bool = True):
    filename = path.expanduser(path.expandvars(filename))
    header: Dict[str, str] = {}
    with open(filename, "rb") as f:
        assert f.readline().decode() == "BEGIN_HEADER\n"
        buffer = f.readline().decode()
        while buffer != "END_HEADER\n":
            key, val = buffer.split("=")
            header[key.strip()] = val.strip()
            buffer = f.readline().decode()
        offset = f.tell()
    latt_size = [
        int(header["DIMENSION_1"]),
        int(header["DIMENSION_2"]),
        int(header["DIMENSION_3"]),
        int(header["DIMENSION_4"]),
    ]
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size)
    assert header["FLOATING_POINT"].startswith("IEEE")
    if header["FLOATING_POINT"][6:] == "BIG":
        endian = ">"
    elif header["FLOATING_POINT"][6:] == "LITTLE":
        endian = "<"
    else:
        raise ValueError(f"Unsupported endian: {header['FLOATING_POINT'][6:]}")
    nbytes = int(header["FLOATING_POINT"][4:6]) // 8
    dtype = f"{endian}c{2 * nbytes}"
    plaquette = float(header["PLAQUETTE"])

    if header["DATATYPE"] == "4D_SU3_GAUGE_3x3":
        gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0))
        if link_trace:
            assert numpy.isclose(
                numpy.einsum("tzyxdaa->", gauge.real) / (gauge.size // Nc), float(header["LINK_TRACE"])
            ), f"Bad link trace for {filename}"
        if checksum:
            assert checksum_nersc(gauge.astype(f"<c{2 * nbytes}").reshape(-1)) == int(
                header["CHECKSUM"], 16
            ), f"Bad checksum for {filename}"
        gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
        return latt_size, plaquette, gauge
    elif header["DATATYPE"] == "4D_SU3_GAUGE":
        # gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc - 1, Nc), (3, 2, 1, 0))
        # if checksum:
        #     assert (
        #         hex(checksum_nersc(gauge.astype(f"<c{2 * nbytes}").reshape(-1)))[2:] == header["CHECKSUM"]
        #     ), f"Bad checksum for {filename}"
        # gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
        # return latt_size, gauge
        raise NotImplementedError("SU3_GAUGE is not supported")
    else:
        raise ValueError(f"Unsupported datatype: {header['DATATYPE']}")
