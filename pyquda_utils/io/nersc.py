from datetime import datetime
from os import path, uname
from typing import Dict, List

import numpy
from mpi4py import MPI

from .mpi_file import getSublatticeSize, readMPIFile, writeMPIFile
from .gauge_utils import gaugeLexicoPlaquette

Nd, Ns, Nc = 4, 4, 3


def checksum_nersc(data: numpy.ndarray) -> int:
    return MPI.COMM_WORLD.allreduce(numpy.sum(data.view("<u4"), dtype="<u4"), MPI.SUM)


def link_trace_nersc(gauge: numpy.ndarray) -> float:
    return MPI.COMM_WORLD.allreduce(
        numpy.einsum("tzyxdaa->", gauge.real) / (MPI.COMM_WORLD.Get_size() * gauge.size // Nc), MPI.SUM
    )


def readGauge(
    filename: str, grid_size: List[int], plaquette: bool = True, link_trace: bool = True, checksum: bool = True
):
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
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    assert header["FLOATING_POINT"].startswith("IEEE")
    if header["FLOATING_POINT"][6:] == "BIG":
        endian = ">"
    elif header["FLOATING_POINT"][6:] == "LITTLE":
        endian = "<"
    else:
        raise ValueError(f"Unsupported endian: {header['FLOATING_POINT'][6:]}")
    float_nbytes = int(header["FLOATING_POINT"][4:6]) // 8
    dtype = f"{endian}c{2 * float_nbytes}"

    if header["DATATYPE"] == "4D_SU3_GAUGE_3x3":
        gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0), grid_size)
        gauge = gauge.astype(f"<c{2 * float_nbytes}")
        if link_trace:
            assert numpy.isclose(link_trace_nersc(gauge), float(header["LINK_TRACE"])), f"Bad link trace for {filename}"
        if checksum:
            assert checksum_nersc(gauge.reshape(-1)) == int(header["CHECKSUM"], 16), f"Bad checksum for {filename}"
        gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
        if plaquette:
            assert numpy.isclose(
                gaugeLexicoPlaquette(latt_size, grid_size, gauge)[0], float(header["PLAQUETTE"])
            ), f"Bad plaquette for {filename}"
        return latt_size, gauge
    elif header["DATATYPE"] == "4D_SU3_GAUGE":
        # gauge = readMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc - 1, Nc), (3, 2, 1, 0))
        # if checksum:
        #     assert checksum_nersc(gauge.astype(f"<c{2 * nbytes}").reshape(-1)) == int(
        #         header["CHECKSUM"], 16
        #     ), f"Bad checksum for {filename}"
        # gauge = gauge.transpose(4, 0, 1, 2, 3, 5, 6).astype("<c16")
        # return latt_size, gauge
        raise NotImplementedError("SU3_GAUGE is not supported")
    else:
        raise ValueError(f"Unsupported datatype: {header['DATATYPE']}")


def writeGauge(
    filename: str,
    latt_size: List[int],
    grid_size: List[int],
    gauge: numpy.ndarray,
    plaquette: float = None,
    use_fp32: bool = False,
):
    filename = path.expanduser(path.expandvars(filename))
    float_nbytes = 4 if use_fp32 else 8
    dtype, offset = f"<c{2 * float_nbytes}", None
    if plaquette is None:
        plaquette = gaugeLexicoPlaquette(latt_size, grid_size, gauge)[0]
    gauge = numpy.ascontiguousarray(gauge.transpose(1, 2, 3, 4, 0, 5, 6).astype(dtype))
    link_trace = link_trace_nersc(gauge)
    checksum = checksum_nersc(gauge.reshape(-1))
    timestamp = datetime.now().astimezone().strftime(R"%a %b %d %H:%M:%S %Y %Z")
    Lx, Ly, Lz, Lt = getSublatticeSize(latt_size, grid_size)
    header: Dict[str, str] = {
        "HDR_VERSION": "1.0",
        "DATATYPE": "4D_SU3_GAUGE_3x3",
        "STORAGE_FORMAT": "",
        "DIMENSION_1": f"{latt_size[0]}",
        "DIMENSION_2": f"{latt_size[1]}",
        "DIMENSION_3": f"{latt_size[2]}",
        "DIMENSION_4": f"{latt_size[3]}",
        "LINK_TRACE": f"{link_trace:.10g}",
        "PLAQUETTE": f"{plaquette:.10g}",
        "BOUNDARY_1": "PERIODIC",
        "BOUNDARY_2": "PERIODIC",
        "BOUNDARY_3": "PERIODIC",
        "BOUNDARY_4": "PERIODIC",
        "CHECKSUM": f"{checksum:10x}",
        "SCIDAC_CHECKSUMA": f"{0:10x}",
        "SCIDAC_CHECKSUMB": f"{0:10x}",
        "ENSEMBLE_ID": "pyquda",
        "ENSEMBLE_LABEL": "",
        "SEQUENCE_NUMBER": "1",
        "CREATOR": "pyquda",
        "CREATOR_HARDWARE": f"{uname().nodename}-{uname().machine}-{uname().sysname}-{uname().release}",
        "CREATION_DATE": timestamp,
        "ARCHIVE_DATE": timestamp,
        "FLOATING_POINT": f"IEEE{float_nbytes * 8}LITTLE",
    }
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(filename, "wb") as f:
            f.write(b"BEGIN_HEADER\n")
            for key, val in header.items():
                f.write(f"{key} = {val}\n".encode())
            f.write(b"END_HEADER\n")
            offset = f.tell()
    offset = MPI.COMM_WORLD.bcast(offset)

    writeMPIFile(filename, dtype, offset, (Lt, Lz, Ly, Lx, Nd, Nc, Nc), (3, 2, 1, 0), grid_size, gauge)
