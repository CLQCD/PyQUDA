import io
import re
import struct
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import numpy as np

from .core import Nc, Nd, Ns, LatticeGauge


def prod(a):
    p = 1
    for i in a:
        p *= i
    return p


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
    latt_size = [
        int(format.find(f"{tag}lx").text),
        int(format.find(f"{tag}ly").text),
        int(format.find(f"{tag}lz").text),
        int(format.find(f"{tag}lt").text),
    ]
    Lx, Ly, Lz, Lt = latt_size
    gauge_raw = np.frombuffer(binary_data, binary_dtype).astype("<c16").reshape(Lt, Lz, Ly, Lx, Nd, Nc, Nc)
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

    return LatticeGauge(latt_size, gauge)


def readIldgBin(filename: str, dtype: str, latt_size: List[int]):
    Lx, Ly, Lz, Lt = latt_size
    gauge_raw = np.fromfile(filename, dtype).astype("<c16").reshape(Lt, Lz, Ly, Lx, Nd, Nc, Nc)
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

    return LatticeGauge(latt_size, gauge)
