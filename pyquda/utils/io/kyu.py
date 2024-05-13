from os import path

import numpy

from ...field import Ns, Nc, Nd, LatticeInfo, LatticeGauge, LatticePropagator, cb2


def fromGaugeBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from ... import openMPIFileRead, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileRead(filename)
    gauge_raw = numpy.empty((Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx), native_dtype)
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Nd, Nc, Nc, 2, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Read_all(gauge_raw)
    filetype.Free()
    fh.Close()

    gauge_raw = (
        gauge_raw.transpose(0, 4, 5, 6, 7, 2, 1, 3)
        .view(dtype)
        .astype("<f8")
        .copy()
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2)
        .view("<c16")
    )

    return gauge_raw


def toGaugeBuffer(filename: str, offset: int, gauge_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from ... import openMPIFileWrite, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileWrite(filename)
    gauge_raw = (
        gauge_raw.view("<f8")
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc, 2)
        .astype(dtype)
        .view(native_dtype)
        .transpose(0, 6, 5, 7, 1, 2, 3, 4)
        .copy()
    )
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Nd, Nc, Nc, 2, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Write_all(gauge_raw)
    filetype.Free()
    fh.Close()


def readGauge(filename: str, latt_info: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    gauge_raw = fromGaugeBuffer(filename, 0, ">f8", latt_info)

    return LatticeGauge(latt_info, cb2(gauge_raw, [1, 2, 3, 4]))


def writeGauge(filename: str, gauge: LatticeGauge):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = gauge.latt_info

    toGaugeBuffer(filename, 0, gauge.lexico(), ">f8", latt_info)


# matrices to convert gamma basis bewteen DeGrand-Rossi and Dirac-Pauli
# \psi(DP) = _DR_TO_DP \psi(DR)
# \psi(DR) = _DP_TO_DR \psi(DP)
_DP_TO_DR = [
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
    [0, 1, 0, 1],
    [-1, 0, -1, 0],
]
_DR_TO_DP = [
    [0, -1, 0, -1],
    [1, 0, 1, 0],
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
]


def rotateToDiracPauli(propagator: LatticePropagator):
    from opt_einsum import contract

    if propagator.location == "numpy":
        A = numpy.asarray(_DP_TO_DR)
        Ainv = numpy.asarray(_DR_TO_DP)
    elif propagator.location == "cupy":
        import cupy

        A = cupy.asarray(_DP_TO_DR)
        Ainv = cupy.asarray(_DR_TO_DP)
    elif propagator.location == "torch":
        import torch

        A = torch.as_tensor(_DP_TO_DR)
        Ainv = torch.as_tensor(_DR_TO_DP)

    _data = contract("ij,etzyxjkab,kl->etzyxilab", Ainv, propagator.data, A, optimize=True)
    return LatticePropagator(propagator.latt_info, _data / 2)


def rotateToDeGrandRossi(propagator: LatticePropagator):
    from opt_einsum import contract

    if propagator.location == "numpy":
        A = numpy.asarray(_DR_TO_DP)
        Ainv = numpy.asarray(_DP_TO_DR)
    elif propagator.location == "cupy":
        import cupy

        A = cupy.array(_DR_TO_DP)
        Ainv = cupy.array(_DP_TO_DR)
    elif propagator.location == "torch":
        import torch

        A = torch.as_tensor(_DR_TO_DP)
        Ainv = torch.as_tensor(_DP_TO_DR)

    _data = contract("ij,etzyxjkab,kl->etzyxilab", Ainv, propagator.data, A, optimize=True)
    return LatticePropagator(propagator.latt_info, _data / 2)


def fromPropagatorBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from ... import openMPIFileRead, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileRead(filename)
    propagator_raw = numpy.empty((Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx), native_dtype)
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Ns, Nc, 2, Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Read_all(propagator_raw)
    filetype.Free()
    fh.Close()

    propagator_raw = (
        propagator_raw.transpose(5, 6, 7, 8, 3, 0, 4, 1, 2)
        .view(dtype)
        .astype("<f8")
        .copy()
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc * 2)
        .view("<c16")
    )

    return propagator_raw


def toPropagatorBuffer(filename: str, offset: int, propagator_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from ... import openMPIFileWrite, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileWrite(filename)
    propagator_raw = (
        propagator_raw.view("<f8")
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc, 2)
        .astype(dtype)
        .view(native_dtype)
        .transpose(5, 7, 8, 4, 6, 0, 1, 2, 3)
        .copy()
    )
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Ns, Nc, 2, Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Write_all(propagator_raw)
    filetype.Free()
    fh.Close()


def readPropagator(filename: str, latt_info: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    propagator_raw = fromPropagatorBuffer(filename, 0, ">f8", latt_info)

    return rotateToDeGrandRossi(LatticePropagator(latt_info, cb2(propagator_raw, [0, 1, 2, 3])))


def writePropagator(filename: str, propagator: LatticePropagator):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = propagator.latt_info

    toPropagatorBuffer(filename, 0, rotateToDiracPauli(propagator).lexico(), ">f8", latt_info)
