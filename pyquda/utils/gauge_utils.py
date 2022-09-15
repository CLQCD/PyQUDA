import io
import re
import struct
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import cupy as cp

from .. import mpi
from ..core import Nc, Nd, cb2, LatticeGauge


def readIldg(filename: str):
    with open(filename, "rb") as f:
        meta: Dict[str, Tuple[int]] = {}
        buffer = f.read(8)
        while buffer != b"":
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
    ndarray_dtype = f"<c{2*precision//8}"
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

    gauge_raw = np.frombuffer(
        binary_data,
        binary_dtype,
    ).reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc,
              Nc)[gt * Lt:(gt + 1) * Lt, gz * Lz:(gz + 1) * Lz, gy * Ly:(gy + 1) * Ly,
                  gx * Lx:(gx + 1) * Lx].astype(ndarray_dtype).transpose(4, 0, 1, 2, 3, 5, 6)

    gauge = cb2(gauge_raw, [1, 2, 3, 4])

    return LatticeGauge(latt_size, gauge, gt == Gt - 1)


def readIldgBin(filename: str, dtype: str, latt_size: List[int]):
    Lx, Ly, Lz, Lt = latt_size
    Gx, Gy, Gz, Gt = mpi.grid
    gx, gy, gz, gt = mpi.coord

    gauge_raw = np.fromfile(
        filename,
        dtype,
    ).reshape(Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc,
              Nc)[gt * Lt:(gt + 1) * Lt, gz * Lz:(gz + 1) * Lz, gy * Ly:(gy + 1) * Ly,
                  gx * Lx:(gx + 1) * Lx].astype("<c16").transpose(4, 0, 1, 2, 3, 5, 6)

    gauge = cb2(gauge_raw, [1, 2, 3, 4])

    return LatticeGauge(latt_size, cp.array(gauge), gt == Gt - 1)
