import io
import re
import struct
import warnings
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import numpy

from .. import mpi
from ..field import Nc, Nd, cb2, LatticeGauge


def readIldg(filename: str):
    """Preserve for compability."""
    warnings.warn("Deprecated. Use `pyquda.utils.io.readQIOGauge` instead.", DeprecationWarning)
    return readQIO(filename)


def readIldgBin(filename: str, dtype: str, latt_size: List[int]):
    """Preserve for compability."""
    warnings.warn("Deprecated. Use `pyquda.utils.io.readILDGBinGauge` instead.", DeprecationWarning)
    return readILDGBin(filename, dtype, latt_size)


def readQIO(filename: str):
    warnings.warn("Deprecated. Use `pyquda.utils.io.readQIOGauge` instead.", DeprecationWarning)
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

        f.seek(meta["ildg-format"][0])
        format = ET.ElementTree(ET.fromstring(f.read(meta["ildg-format"][1]).strip(b"\x00").decode("utf-8")))
        f.seek(meta["ildg-binary-data"][0])
        binary_data = f.read(meta["ildg-binary-data"][1])
    tag = re.match(r"\{.*\}", format.getroot().tag).group(0)
    precision = int(format.find(f"{tag}precision").text)
    binary_dtype = f">c{2*precision//8}"
    ndarray_dtype = "<c16"
    latt_size = [
        int(format.find(f"{tag}lx").text),
        int(format.find(f"{tag}ly").text),
        int(format.find(f"{tag}lz").text),
        int(format.find(f"{tag}lt").text),
    ]
    Lx, Ly, Lz, Lt = latt_size
    Gx, Gy, Gz, Gt = mpi.grid
    gx, gy, gz, gt = mpi.coord
    latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
    Lx, Ly, Lz, Lt = latt_size

    gauge_raw = (
        numpy.frombuffer(binary_data, binary_dtype)
        .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc, Nc)[
            gt * Lt : (gt + 1) * Lt, gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx
        ]
        .astype(ndarray_dtype)
        .transpose(4, 0, 1, 2, 3, 5, 6)
    )

    gauge = cb2(gauge_raw, [1, 2, 3, 4])

    return LatticeGauge(latt_size, gauge)


def readILDGBin(filename: str, dtype: str, latt_size: List[int]):
    warnings.warn("Deprecated. Use `pyquda.utils.io.readILDGBinGauge` instead.", DeprecationWarning)
    Lx, Ly, Lz, Lt = latt_size
    Gx, Gy, Gz, Gt = mpi.grid
    gx, gy, gz, gt = mpi.coord

    gauge_raw = (
        numpy.fromfile(filename, dtype)
        .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc, Nc)[
            gt * Lt : (gt + 1) * Lt, gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx
        ]
        .astype("<c16")
        .transpose(4, 0, 1, 2, 3, 5, 6)
    )

    gauge = cb2(gauge_raw, [1, 2, 3, 4])

    return LatticeGauge(latt_size, gauge)


def readMILC(filename: str):
    warnings.warn("Deprecated. Use `pyquda.utils.io.readMILCGauge` instead.", DeprecationWarning)
    with open(filename, "rb") as f:
        magic = f.read(4)
        assert struct.unpack("<i", magic)[0] == 20103
        latt_size = struct.unpack("<iiii", f.read(16))
        Lx, Ly, Lz, Lt = latt_size
        time_stamp = f.read(64).decode()
        assert struct.unpack("<i", f.read(4))[0] == 0
        sum29, sum31 = struct.unpack("<II", f.read(8))
        binary_data = f.read(Lt * Lz * Ly * Lx * Nd * Nc * Nc * 2 * 4)
    binary_dtype = "<c8"
    ndarray_dtype = "<c16"
    Lx, Ly, Lz, Lt = latt_size
    Gx, Gy, Gz, Gt = mpi.grid
    gx, gy, gz, gt = mpi.coord
    latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
    Lx, Ly, Lz, Lt = latt_size

    gauge_raw = (
        numpy.frombuffer(binary_data, binary_dtype)
        .reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc, Nc)[
            gt * Lt : (gt + 1) * Lt, gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx
        ]
        .astype(ndarray_dtype)
        .transpose(4, 0, 1, 2, 3, 5, 6)
    )

    gauge = cb2(gauge_raw, [1, 2, 3, 4])

    return LatticeGauge(latt_size, gauge)


def unitGauge(latt_size: List[int]):
    gauge = LatticeGauge(latt_size, None)

    return gauge


def gaussGauge(latt_size: List[int], seed: int):
    from ..pyquda import loadGaugeQuda, saveGaugeQuda, gaussGaugeQuda
    from ..core import getDslash

    gauge = LatticeGauge(latt_size, None)

    dslash = getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
    dslash.gauge_param.use_resident_gauge = 0
    loadGaugeQuda(gauge.data_ptrs, dslash.gauge_param)
    dslash.gauge_param.use_resident_gauge = 1
    gaussGaugeQuda(seed, 1.0)
    saveGaugeQuda(gauge.data_ptrs, dslash.gauge_param)

    return gauge
