import io
import re
import struct
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import cupy as cp

from ..core import Nc, Nd, LatticeGauge


def readIldg(filename: str, grid_size: List[int] = None, rank: int = 0):
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
    latt_size = [
        int(format.find(f"{tag}lx").text),
        int(format.find(f"{tag}ly").text),
        int(format.find(f"{tag}lz").text),
        int(format.find(f"{tag}lt").text),
    ]
    Lx, Ly, Lz, Lt = latt_size
    gauge_raw = np.frombuffer(binary_data, binary_dtype).astype("<c16").reshape(Lt, Lz, Ly, Lx, Nd, Nc, Nc)

    if grid_size is not None:
        Gx, Gy, Gz, Gt = grid_size
        assert rank < Gx * Gy * Gz * Gt
        latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
        Lx, Ly, Lz, Lt = latt_size
        gt = rank % Gt
        gz = rank // Gt % Gz
        gy = rank // Gt // Gz % Gy
        gx = rank // Gt // Gz // Gy
        gauge_raw = gauge_raw[gt * Lt:(gt + 1) * Lt, gz * Lz:(gz + 1) * Lz, gy * Ly:(gy + 1) * Ly,
                              gx * Lx:(gx + 1) * Lx]
    else:
        Gt = 1
        gt = 0

    gauge = np.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")

    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    gauge[:, 0, t, z, y, :, :, :] = gauge_raw[t, z, y, 0::2, :, :, :].transpose(1, 0, 2, 3)
                    gauge[:, 1, t, z, y, :, :, :] = gauge_raw[t, z, y, 1::2, :, :, :].transpose(1, 0, 2, 3)
                else:
                    gauge[:, 0, t, z, y, :, :, :] = gauge_raw[t, z, y, 1::2, :, :, :].transpose(1, 0, 2, 3)
                    gauge[:, 1, t, z, y, :, :, :] = gauge_raw[t, z, y, 0::2, :, :, :].transpose(1, 0, 2, 3)

    return LatticeGauge(latt_size, cp.array(gauge), gt == Gt - 1)


def readIldgBin(filename: str, dtype: str, latt_size: List[int], grid_size: List[int] = None, rank: int = 0):
    Lx, Ly, Lz, Lt = latt_size
    gauge_raw = np.fromfile(filename, dtype).astype("<c16").reshape(Lt, Lz, Ly, Lx, Nd, Nc, Nc)

    if grid_size is not None:
        Gx, Gy, Gz, Gt = grid_size
        latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
        Lx, Ly, Lz, Lt = latt_size
        assert rank < Gx * Gy * Gz * Gt
        gt = rank % Gt
        gz = rank // Gt % Gz
        gy = rank // Gt // Gz % Gy
        gx = rank // Gt // Gz // Gy
        gauge_raw = gauge_raw[gt * Lt:(gt + 1) * Lt, gz * Lz:(gz + 1) * Lz, gy * Ly:(gy + 1) * Ly,
                              gx * Lx:(gx + 1) * Lx]
    else:
        Gt = 1
        gt = 0

    gauge = np.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")

    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    gauge[:, 0, t, z, y, :, :, :] = gauge_raw[t, z, y, 0::2, :, :, :].transpose(1, 0, 2, 3)
                    gauge[:, 1, t, z, y, :, :, :] = gauge_raw[t, z, y, 1::2, :, :, :].transpose(1, 0, 2, 3)
                else:
                    gauge[:, 0, t, z, y, :, :, :] = gauge_raw[t, z, y, 1::2, :, :, :].transpose(1, 0, 2, 3)
                    gauge[:, 1, t, z, y, :, :, :] = gauge_raw[t, z, y, 0::2, :, :, :].transpose(1, 0, 2, 3)

    return LatticeGauge(latt_size, cp.array(gauge), gt == Gt - 1)
