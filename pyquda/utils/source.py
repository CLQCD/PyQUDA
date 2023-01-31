from typing import List, Union

from .. import mpi
from ..field import Nc, Ns, LatticeFermion, LatticePropagator


def point(latt_size: List[int], t_srce: List[int], spin: int, color: int):
    Lx, Ly, Lz, Lt = latt_size
    x, y, z, t = t_srce
    gx, gy, gz, gt = mpi.coord
    b = LatticeFermion(latt_size)
    data = b.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    if (
        gx * Lx <= x < (gx + 1) * Lx and gy * Ly <= y < (gy + 1) * Ly and gz * Lz <= z < (gz + 1) * Lz and
        gt * Lt <= t < (gt + 1) * Lt
    ):
        eo = ((x - gx * Lx) + (y - gy * Ly) + (z - gz * Lz) + (t - gt * Lt)) % 2
        data[eo, t - gt * Lt, z - gz * Lz, y - gy * Ly, (x - gx * Lx) // 2, spin, color] = 1

    return b


def wall(latt_size: List[int], t_srce: int, spin: int, color: int):
    Lx, Ly, Lz, Lt = latt_size
    gx, gy, gz, gt = mpi.coord
    t = t_srce
    b = LatticeFermion(latt_size)
    data = b.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    if gt * Lt <= t < (gt + 1) * Lt:
        data[:, t - gt * Lt, :, :, :, spin, color] = 1

    return b


def momentum(latt_size: List[int], t_srce: int, phase, spin: int, color: int):
    Lx, Ly, Lz, Lt = latt_size
    gx, gy, gz, gt = mpi.coord
    t = t_srce
    b = LatticeFermion(latt_size)
    data = b.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    if gt * Lt <= t < (gt + 1) * Lt:
        data[:, t - gt * Lt, :, :, :, spin, color] = phase[:, t - gt * Lt, :, :, :]

    return b


def colorvec(latt_size: List[int], t_srce: int, phase, spin: int):
    Lx, Ly, Lz, Lt = latt_size
    gx, gy, gz, gt = mpi.coord
    t = t_srce
    b = LatticeFermion(latt_size)
    data = b.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    if gt * Lt <= t < (gt + 1) * Lt:
        data[:, t - gt * Lt, :, :, :, spin, :] = phase[:, t - gt * Lt, :, :, :, :]
    return b


def source(latt_size: List[int], source_type: str, t_srce: Union[int, List[int]], spin: int, color: int, phase=None):
    if source_type.lower() == "point":
        return point(latt_size, t_srce, spin, color)
    elif source_type.lower() == "wall":
        return wall(latt_size, t_srce, spin, color)
    elif source_type.lower() == "momentum":
        return momentum(latt_size, t_srce, phase, spin, color)
    elif source_type.lower() == "colorvec":
        return colorvec(latt_size, t_srce, phase, spin)
    else:
        raise NotImplementedError(f"{source_type} source is not implemented yet.")


def source12(latt_size: List[int], source_type: str, t_srce: Union[int, List[int]], phase=None):
    Lx, Ly, Lz, Lt = latt_size
    Vol = Lx * Ly * Lz * Lt

    b12 = LatticePropagator(latt_size)
    data = b12.data.reshape(Vol, Ns, Ns, Nc, Nc)
    for spin in range(Ns):
        for color in range(Nc):
            b = source(latt_size, source_type, t_srce, spin, color, phase)
            data[:, :, spin, :, color] = b.data.reshape(Vol, Ns, Nc)

    return b12
