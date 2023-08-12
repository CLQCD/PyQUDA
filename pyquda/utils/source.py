from typing import List, Literal, Union

from .. import mpi
from ..field import Nc, Ns, LatticeColorVector, LatticeFermion, LatticePropagator


def point(latt_size: List[int], t_srce: List[int], spin: int, color: int):
    Lx, Ly, Lz, Lt = latt_size
    x, y, z, t = t_srce
    gx, gy, gz, gt = mpi.coord
    b = LatticeFermion(latt_size)
    data = b.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
    if (
        gx * Lx <= x < (gx + 1) * Lx
        and gy * Ly <= y < (gy + 1) * Ly
        and gz * Lz <= z < (gz + 1) * Lz
        and gt * Lt <= t < (gt + 1) * Lt
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


def gaussian(latt_size: List[int], t_srce: int, color: int, rho: float, nsteps: int, xi: float):
    from .. import core
    from ..enum_quda import QudaDslashType, QudaParity

    def _Laplacian(src, aux, sigma, xi, invert_param):
        aux.data[:] = 0
        core.quda.dslashQuda(aux.even_ptr, src.odd_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY)
        core.quda.dslashQuda(aux.odd_ptr, src.even_ptr, invert_param, QudaParity.QUDA_ODD_PARITY)
        aux.even -= src.odd
        aux.odd -= src.even
        aux.data *= xi
        src.data = (1 - sigma * 6) * src.data + sigma * aux.data

    Lx, Ly, Lz, Lt = latt_size
    gx, gy, gz, gt = mpi.coord
    x, y, z, t = t_srce
    b = LatticeColorVector(latt_size)
    data = b.data.reshape(2, Lt, Lz, Ly, Lx // 2, Nc)

    if (
        gx * Lx <= x < (gx + 1) * Lx
        and gy * Ly <= y < (gy + 1) * Ly
        and gz * Lz <= z < (gz + 1) * Lz
        and gt * Lt <= t < (gt + 1) * Lt
    ):
        eo = ((x - gx * Lx) + (y - gy * Ly) + (z - gz * Lz) + (t - gt * Lt)) % 2
        data[eo, t - gt * Lt, z - gz * Lz, y - gy * Ly, (x - gx * Lx) // 2, color] = 1

    dslash = core.getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
    dslash.invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
    c = core.LatticeColorVector(latt_size)
    for _ in range(nsteps):
        # (rho**2 / 4) aims to achieve the same result with Chroma
        _Laplacian(b, c, rho**2 / 4 / nsteps, xi, dslash.invert_param)

    return b


def colorvector(latt_size: List[int], t_srce: int, phase):
    Lx, Ly, Lz, Lt = latt_size
    gx, gy, gz, gt = mpi.coord
    t = t_srce
    b = LatticeColorVector(latt_size)
    data = b.data.reshape(2, Lt, Lz, Ly, Lx // 2, Nc)
    if gt * Lt <= t < (gt + 1) * Lt:
        data[:, t - gt * Lt, :, :, :, :] = phase[:, t - gt * Lt, :, :, :, :]
    return b


def source(
    latt_size: List[int],
    source_type: str,
    t_srce: Union[int, List[int]],
    spin: int,
    color: int,
    phase=None,
    rho: float = 0.0,
    nsteps: int = 0,
    xi: float = 1.0,
):
    if source_type.lower() == "point":
        return point(latt_size, t_srce, spin, color)
    elif source_type.lower() == "wall":
        return wall(latt_size, t_srce, spin, color)
    elif source_type.lower() == "momentum":
        return momentum(latt_size, t_srce, phase, spin, color)
    elif source_type.lower() == "gaussian":
        return gaussian(latt_size, t_srce, color, rho, nsteps, xi)
    elif source_type.lower() == "colorvector":
        return colorvector(latt_size, t_srce, phase)
    else:
        raise NotImplementedError(f"{source_type} source is not implemented yet.")


def source12(
    latt_size: List[int],
    source_type: Literal["point", "wall", "momentum", "gaussian", "colorvector"],
    t_srce: Union[int, List[int]],
    phase=None,
    rho: float = 0.0,
    nsteps: int = 0,
    xi: float = 1.0,
):
    Lx, Ly, Lz, Lt = latt_size
    Vol = Lx * Ly * Lz * Lt

    b12 = LatticePropagator(latt_size)
    data = b12.data.reshape(Vol, Ns, Ns, Nc, Nc)
    if source_type.lower() in ["colorvector"]:
        b = source(latt_size, source_type, t_srce, 0, 0, phase)
        for color in range(Nc):
            for spin in range(Ns):
                data[:, spin, spin, :, color] = b.data.reshape(Vol, Nc)
    elif source_type.lower() in ["gaussian"]:
        for color in range(Nc):
            b = source(latt_size, source_type, t_srce, 0, color, phase, rho, nsteps, xi)
            for spin in range(Ns):
                data[:, spin, spin, :, color] = b.data.reshape(Vol, Nc)
    else:
        for color in range(Nc):
            for spin in range(Ns):
                b = source(latt_size, source_type, t_srce, spin, color, phase)
                data[:, :, spin, :, color] = b.data.reshape(Vol, Ns, Nc)

    return b12
