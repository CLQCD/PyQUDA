from typing import List, Literal, Union

from pyquda import getGridSize, getLogger
from pyquda.field import (
    Ns,
    Nc,
    LatticeInfo,
    LatticeGauge,
    LatticeFermion,
    MultiLatticeFermion,
    LatticePropagator,
    LatticeStaggeredFermion,
    MultiLatticeStaggeredFermion,
    LatticeStaggeredPropagator,
)
from pyquda.enum_quda import QudaDslashType, QudaParity


def point(latt_info: LatticeInfo, t_srce: List[int], spin: Union[int, None], color: int, phase=None):
    Lx, Ly, Lz, Lt = latt_info.size
    gx, gy, gz, gt = latt_info.grid_coord
    x, y, z, t = t_srce
    b = LatticeFermion(latt_info) if spin is not None else LatticeStaggeredFermion(latt_info)
    if (
        gx * Lx <= x < (gx + 1) * Lx
        and gy * Ly <= y < (gy + 1) * Ly
        and gz * Lz <= z < (gz + 1) * Lz
        and gt * Lt <= t < (gt + 1) * Lt
    ):
        eo = ((x - gx * Lx) + (y - gy * Ly) + (z - gz * Lz) + (t - gt * Lt)) % 2
        if spin is not None:
            b.data[eo, t - gt * Lt, z - gz * Lz, y - gy * Ly, (x - gx * Lx) // 2, spin, color] = (
                phase[eo, t - gt * Lt, z - gz * Lz, y - gy * Ly, (x - gx * Lx) // 2] if phase is not None else 1
            )
        else:
            b.data[eo, t - gt * Lt, z - gz * Lz, y - gy * Ly, (x - gx * Lx) // 2, color] = (
                phase[eo, t - gt * Lt, z - gz * Lz, y - gy * Ly, (x - gx * Lx) // 2] if phase is not None else 1
            )

    return b


def wall(latt_info: LatticeInfo, t_srce: int, spin: Union[int, None], color: int, phase=None):
    Lt = latt_info.Lt
    gt = latt_info.gt
    t = t_srce
    b = LatticeFermion(latt_info) if spin is not None else LatticeStaggeredFermion(latt_info)
    if gt * Lt <= t < (gt + 1) * Lt:
        if spin is not None:
            b.data[:, t - gt * Lt, :, :, :, spin, color] = phase[:, t - gt * Lt] if phase is not None else 1
        else:
            b.data[:, t - gt * Lt, :, :, :, color] = phase[:, t - gt * Lt] if phase is not None else 1

    return b


def volume(latt_info: LatticeInfo, spin: Union[int, None], color: int, phase=None):
    b = LatticeFermion(latt_info) if spin is not None else LatticeStaggeredFermion(latt_info)
    if spin is not None:
        b.data[:, :, :, :, :, spin, color] = phase if phase is not None else 1
    else:
        b.data[:, :, :, :, :, color] = phase if phase is not None else 1

    return b


def fermion(
    latt_info: LatticeInfo,
    source_type: Literal["point", "wall", "volume", "momentum", "colorvector"],
    t_srce: Union[List[int], int, None],
    spin: Union[int, None],
    color: int,
    source_phase=None,
):
    if source_type.lower() == "point":
        return point(latt_info, t_srce, spin, color, source_phase)
    elif source_type.lower() == "wall":
        return wall(latt_info, t_srce, spin, color, source_phase)
    elif source_type.lower() == "volume":
        return volume(latt_info, spin, color, source_phase)
    else:
        getLogger().critical(f"{source_type} source is not implemented yet", NotImplementedError)


def multiFermion(
    latt_info: LatticeInfo,
    source_type: Literal["point", "wall", "volume"],
    t_srce: Union[List[int], int, None],
    source_phase=None,
):
    b = MultiLatticeFermion(latt_info, Ns * Nc)
    for spin in range(Ns):
        for color in range(Nc):
            b[spin * Nc + color] = source(latt_info, source_type, t_srce, spin, color, source_phase)
    return b


def multiStaggeredFermion(
    latt_info: LatticeInfo,
    source_type: Literal["point", "wall", "volume"],
    t_srce: Union[List[int], int, None],
    source_phase=None,
):
    b = MultiLatticeStaggeredFermion(latt_info, Nc)
    for color in range(Nc):
        b[color] = source(latt_info, source_type, t_srce, None, color, source_phase)
    return b


def propagator(
    latt_info: LatticeInfo,
    source_type: Literal["point", "wall", "volume"],
    t_srce: Union[List[int], int, None],
    source_phase=None,
):
    b = LatticePropagator(latt_info)
    for spin in range(Ns):
        for color in range(Nc):
            b.setFermion(source(latt_info, source_type, t_srce, spin, color, source_phase), spin, color)
    return b


def staggeredPropagator(
    latt_info: LatticeInfo,
    source_type: Literal["point", "wall", "volume"],
    t_srce: Union[List[int], int, None],
    source_phase=None,
):
    b = LatticeStaggeredPropagator(latt_info)
    for color in range(Nc):
        b.setFermion(source(latt_info, source_type, t_srce, None, color, source_phase), color)
    return b


def momentum(latt_info: LatticeInfo, t_srce: Union[int, None], spin: Union[int, None], color: int, phase):
    t = t_srce
    if t is not None:
        b = wall(latt_info, t_srce, spin, color, phase)
    else:
        b = volume(latt_info, spin, color, phase)

    return b


def colorvector(latt_info: LatticeInfo, t_srce: Union[int, None], spin: Union[int, None], phase):
    Lt = latt_info.Lt
    gt = latt_info.gt
    t = t_srce
    b = LatticeFermion(latt_info) if spin is not None else LatticeStaggeredFermion(latt_info)
    if t is not None:
        if gt * Lt <= t < (gt + 1) * Lt:
            if spin is not None:
                b.data[:, t - gt * Lt, :, :, :, spin, :] = phase[:, t - gt * Lt]
            else:
                b.data[:, t - gt * Lt, :, :, :, :] = phase[:, t - gt * Lt]
    else:
        if spin is not None:
            b.data[:, :, :, :, :, spin, :] = phase
        else:
            b.data[:, :, :, :, :, :] = phase

    return b


def source(
    latt_info: Union[LatticeInfo, List[int]],
    source_type: Literal["point", "wall", "volume", "momentum", "colorvector"],
    t_srce: Union[List[int], int, None],
    spin: Union[int, None],
    color: int,
    source_phase=None,
):
    if isinstance(latt_info, LatticeInfo):
        pass
    else:
        getLogger().warning(
            "source(latt_size: List[int]) is deprecated, use source(latt_info: LatticeInfo) instead",
            DeprecationWarning,
        )

        Gx, Gy, Gz, Gt = getGridSize()
        Lx, Ly, Lz, Lt = latt_info
        Lx, Ly, Lz, Lt = Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt
        latt_info = LatticeInfo([Lx, Ly, Lz, Lt])

    if source_type.lower() == "point":
        return point(latt_info, t_srce, spin, color, source_phase)
    elif source_type.lower() == "wall":
        return wall(latt_info, t_srce, spin, color, source_phase)
    elif source_type.lower() == "volume":
        return volume(latt_info, spin, color, source_phase)
    elif source_type.lower() == "momentum":
        return momentum(latt_info, t_srce, spin, color, source_phase)
    elif source_type.lower() == "colorvector":
        return colorvector(latt_info, t_srce, spin, source_phase)
    else:
        getLogger().critical(f"{source_type} source is not implemented yet", NotImplementedError)


def source12(
    latt_info: Union[LatticeInfo, List[int]],
    source_type: Literal["point", "wall", "volume", "momentum", "colorvector"],
    t_srce: Union[List[int], int, None],
    source_phase=None,
):
    if not isinstance(latt_info, LatticeInfo):
        getLogger().warning(
            "source12(latt_size: List[int]) is deprecated, use source12(latt_info: LatticeInfo) instead",
            DeprecationWarning,
        )

        Gx, Gy, Gz, Gt = getGridSize()
        Lx, Ly, Lz, Lt = latt_info
        Lx, Ly, Lz, Lt = Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt
        latt_info = LatticeInfo([Lx, Ly, Lz, Lt])
    volume = latt_info.volume

    b12 = LatticePropagator(latt_info)
    data = b12.data.reshape(volume, Ns, Ns, Nc, Nc)
    if source_type.lower() in ["colorvector"]:
        b = source(latt_info, source_type, t_srce, None, None, source_phase)
        for color in range(Nc):
            for spin in range(Ns):
                data[:, spin, spin, :, color] = b.data.reshape(volume, Nc)
    else:
        for color in range(Nc):
            for spin in range(Ns):
                b = source(latt_info, source_type, t_srce, spin, color, source_phase)
                data[:, :, spin, :, color] = b.data.reshape(volume, Ns, Nc)

    return b12


def source3(
    latt_info: Union[LatticeInfo, List[int]],
    source_type: Literal["point", "wall", "volume", "momentum", "colorvector"],
    t_srce: Union[List[int], int, None],
    source_phase=None,
):
    if not isinstance(latt_info, LatticeInfo):
        getLogger().warning(
            "source3(latt_size: List[int]) is deprecated, use source3(latt_info: LatticeInfo) instead",
            DeprecationWarning,
        )

        Gx, Gy, Gz, Gt = getGridSize()
        Lx, Ly, Lz, Lt = latt_info
        Lx, Ly, Lz, Lt = Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt
        latt_info = LatticeInfo([Lx, Ly, Lz, Lt])
    volume = latt_info.volume

    b3 = LatticeStaggeredPropagator(latt_info)
    data = b3.data.reshape(volume, Nc, Nc)

    for color in range(Nc):
        b = source(latt_info, source_type, t_srce, None, color, source_phase)
        data[:, :, color] = b.data.reshape(volume, Nc)

    return b3


def gaussianSmear(
    x: Union[
        LatticeFermion,
        LatticeStaggeredPropagator,
        MultiLatticeFermion,
        MultiLatticeStaggeredFermion,
        LatticePropagator,
        LatticeStaggeredPropagator,
    ],
    gauge: LatticeGauge,
    rho: float,
    n_steps: int,
):
    alpha = 1 / (4 * n_steps / rho**2 - 6)
    gauge.gauge_dirac.loadGauge(gauge)
    if isinstance(x, LatticeFermion) or isinstance(x, LatticeStaggeredFermion):
        b = gauge.gauge_dirac.wuppertalSmear(x, alpha)
    elif isinstance(x, MultiLatticeFermion):
        b = MultiLatticeFermion(x.latt_info, x.L5)
        for index in range(x.L5):
            b[index] = gauge.gauge_dirac.wuppertalSmear(x[index], alpha)
    elif isinstance(x, MultiLatticeStaggeredFermion):
        b = MultiLatticeStaggeredFermion(x.latt_info, x.L5)
        for index in range(x.L5):
            b[index] = gauge.gauge_dirac.wuppertalSmear(x[index], alpha)
    elif isinstance(x, LatticePropagator):
        b = LatticePropagator(x.latt_info)
        for spin in range(Ns):
            for color in range(Nc):
                b.setFermion(gauge.gauge_dirac.wuppertalSmear(x.getFermion(spin, color), n_steps, alpha), spin, color)
    elif isinstance(x, LatticeStaggeredPropagator):
        b = LatticeStaggeredPropagator(x.latt_info)
        for color in range(Nc):
            b.setFermion(gauge.gauge_dirac.wuppertalSmear(x.getFermion(color), n_steps, alpha), color)
    gauge.gauge_dirac.loadGauge(gauge)
    return b


def gaussian(x: Union[LatticeFermion, LatticeStaggeredFermion], gauge: LatticeGauge, rho: float, n_steps: int):
    alpha = 1 / (4 * n_steps / rho**2 - 6)
    return gauge.wuppertalSmear(x, n_steps, alpha)


def gaussian12(x12: LatticePropagator, gauge: LatticeGauge, rho: float, n_steps: int):
    alpha = 1 / (4 * n_steps / rho**2 - 6)
    b12 = LatticePropagator(x12.latt_info)
    gauge.gauge_dirac.loadGauge(gauge)
    for spin in range(Ns):
        for color in range(Nc):
            b12.setFermion(gauge.gauge_dirac.wuppertalSmear(x12.getFermion(spin, color), n_steps, alpha), spin, color)
    gauge.gauge_dirac.freeGauge()

    return b12


def gaussian3(x3: LatticeStaggeredPropagator, gauge: LatticeGauge, rho: float, n_steps: int):
    alpha = 1 / (4 * n_steps / rho**2 - 6)
    b3 = LatticeStaggeredPropagator(x3.latt_info)
    gauge.gauge_dirac.loadGauge(gauge)
    for color in range(Nc):
        b3.setFermion(gauge.gauge_dirac.wuppertalSmear(x3.getFermion(color), n_steps, alpha), color)
    gauge.gauge_dirac.freeGauge()

    return b3


def _gaussian3(latt_info: LatticeInfo, t_srce: List[int], spin: int, color: int, rho: float, n_steps: int):
    from pyquda import pyquda as quda
    from .. import core

    _b = point(latt_info, t_srce, None, color)
    dslash = core.getDslash(latt_info.size, 0, 0, 0, anti_periodic_t=False)
    dslash.invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
    alpha = 1 / (4 * n_steps / rho**2 - 6)
    quda.performWuppertalnStep(_b.data_ptr, _b.data_ptr, dslash.invert_param, n_steps, alpha)

    if spin is not None:
        b = LatticeFermion(latt_info)
        b.data[:, :, :, :, :, spin, :] = _b.data
    else:
        b = LatticeStaggeredFermion(latt_info, _b.data)

    return b


def _gaussian2(latt_info: LatticeInfo, t_srce: List[int], spin: int, color: int, rho: float, n_steps: int, xi: float):
    from pyquda import pyquda as quda
    from .. import core

    def _Laplacian(src, aux, sigma, invert_param):
        # aux = -kappa * Laplace * src + src
        quda.MatQuda(aux.data_ptr, src.data_ptr, invert_param)
        src.data = -6 * sigma * src.data + aux.data

    _b = point(latt_info, t_srce, None, color)
    _c = LatticeStaggeredFermion(latt_info)

    # use mass to get specific kappa = -xi * rho**2 / 4 / n_steps
    kappa = -(rho**2) / 4 / n_steps * xi
    mass = 1 / (2 * kappa) - 4
    dslash = core.getDslash(latt_info.size, mass, 0, 0, anti_periodic_t=False)
    dslash.invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
    for _ in range(n_steps):
        # (rho**2 / 4) here aims to achieve the same result with Chroma
        _Laplacian(_b, _c, rho**2 / 4 / n_steps, dslash.invert_param)
    _c = None

    if spin is not None:
        b = LatticeFermion(latt_info)
        b.data[:, :, :, :, :, spin, :] = _b.data
    else:
        b = LatticeStaggeredFermion(latt_info, _b.data)

    return b


def _gaussian1(latt_info: LatticeInfo, t_srce: List[int], spin: int, color: int, rho: float, n_steps: int, xi: float):
    from pyquda import pyquda as quda
    from .. import core

    def _Laplacian(src, aux, sigma, xi, invert_param):
        aux.data[:] = 0
        quda.dslashQuda(aux.even_ptr, src.odd_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY)
        quda.dslashQuda(aux.odd_ptr, src.even_ptr, invert_param, QudaParity.QUDA_ODD_PARITY)
        aux.even -= src.odd
        aux.odd -= src.even
        aux.data *= xi
        src.data = (1 - sigma * 6) * src.data + sigma * aux.data

    Lx, Ly, Lz, Lt = latt_info.size
    gx, gy, gz, gt = latt_info.grid_coord
    x, y, z, t = t_srce
    _b = LatticeStaggeredFermion(latt_info)
    _c = LatticeStaggeredFermion(latt_info)

    if (
        gx * Lx <= x < (gx + 1) * Lx
        and gy * Ly <= y < (gy + 1) * Ly
        and gz * Lz <= z < (gz + 1) * Lz
        and gt * Lt <= t < (gt + 1) * Lt
    ):
        eo = ((x - gx * Lx) + (y - gy * Ly) + (z - gz * Lz) + (t - gt * Lt)) % 2
        _b.data[eo, t - gt * Lt, z - gz * Lz, y - gy * Ly, (x - gx * Lx) // 2, color] = 1

    dslash = core.getDslash(latt_info.size, 0, 0, 0, anti_periodic_t=False)
    dslash.invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
    for _ in range(n_steps):
        # (rho**2 / 4) aims to achieve the same result with Chroma
        _Laplacian(_b, _c, rho**2 / 4 / n_steps, xi, dslash.invert_param)
    _c = None

    if spin is not None:
        b = LatticeFermion(latt_info)
        b.data[:, :, :, :, :, spin, :] = _b.data
    else:
        b = LatticeStaggeredFermion(latt_info, _b.data)

    return b


def sequential(x: Union[LatticeFermion, LatticeStaggeredFermion], t_srce: int):
    Lt = x.latt_info.Lt
    gt = x.latt_info.gt
    t = t_srce
    if isinstance(x, LatticeStaggeredFermion):
        b = LatticeStaggeredFermion(x.latt_info)
    else:
        b = LatticeFermion(x.latt_info)
    if gt * Lt <= t < (gt + 1) * Lt:
        b.data[:, t - gt * Lt, :, :, :] = x.data[:, t - gt * Lt, :, :, :]

    return b


def sequential12(x12: LatticePropagator, t_srce: int):
    b12 = LatticePropagator(x12.latt_info)
    for spin in range(Ns):
        for color in range(Nc):
            b12.setFermion(sequential(x12.getFermion(spin, color), t_srce), spin, color)

    return b12


def sequential3(x3: LatticeStaggeredPropagator, t_srce: int):
    b3 = LatticeStaggeredPropagator(x3.latt_info)
    for color in range(Nc):
        b3.setFermion(sequential(x3.getFermion(color), t_srce))

    return b3
