from typing import List
import numpy

from .core import evenodd, getGridSize, LatticeGauge, LatticeInfo, LatticePropagator

import gpt as g


def LatticeInfoGPT(grid: g.grid, gen_simd_width: int):
    assert getGridSize() == grid.mpi
    GLx, GLy, GLz, GLt = grid.fdimensions
    Gx, Gy, Gz, Gt = grid.mpi
    Lx, Ly, Lz, Lt = GLx // Gx, GLy // Gy, GLz // Gz, GLt // Gt
    sublatt_size = [Lx, Ly, Lz, Lt]
    Nd = len(sublatt_size)
    precision = grid.precision.nbytes
    n_simd = gen_simd_width // (2 * precision)
    simd = [1] * Nd
    i = Nd - 1
    while n_simd > 1:
        simd[i] *= 2
        n_simd //= 2
        i = i - 1 if i > 0 else Nd - 1
    return LatticeInfo(grid.fdimensions), [sublatt_size[i] // simd[i] for i in range(Nd)], simd, precision


def LatticeGaugeGPT(lattice: List[g.lattice], gen_simd_width: int, gauge: LatticeGauge = None):
    latt_info, gpt_latt, gpt_simd, gpt_prec = LatticeInfoGPT(lattice[0].grid, gen_simd_width)
    Lx, Ly, Lz, Lt = latt_info.size
    Nc = latt_info.Nc
    assert lattice[0].describe().startswith(f"ot_matrix_su_n_fundamental_group({Nc})")
    assert len(lattice) == latt_info.Nd
    if gauge is None:
        value = []
        for index in range(latt_info.Nd):
            value.append(
                evenodd(
                    numpy.asarray(lattice[index].mview()[0])
                    .view(f"<c{2 * gpt_prec}")
                    .reshape(*gpt_latt[::-1], Nc, Nc, *gpt_simd[::-1])
                    .transpose(6, 0, 7, 1, 8, 2, 9, 3, 4, 5)
                    .reshape(Lt, Lz, Ly, Lx, Nc, Nc)
                    .astype("<c16"),
                    [0, 1, 2, 3],
                )
            )
        gauge = LatticeGauge(latt_info, numpy.asarray(value))
        return gauge
    else:
        assert latt_info.size == gauge.latt_info.size
        for index in range(latt_info.Nd):
            gpt_shape = [i for sl in zip(gpt_simd[::-1], gpt_latt[::-1]) for i in sl]
            lattice[index].mview()[0][:] = (
                gauge[index]
                .lexico()
                .astype(f"<c{2 * gpt_prec}")
                .reshape(*gpt_shape, Nc, Nc)
                .transpose(1, 3, 5, 7, 8, 9, 0, 2, 4, 6)
                .copy()  # .view("|u1") requires this
                .view("|u1")
                .reshape(-1)
            )


def LatticePropagatorGPT(lattice: g.lattice, gen_simd_width: int, propagator: LatticePropagator = None):
    latt_info, gpt_latt, gpt_simd, gpt_prec = LatticeInfoGPT(lattice.grid, gen_simd_width)
    Lx, Ly, Lz, Lt = latt_info.size
    Ns, Nc = latt_info.Ns, latt_info.Nc
    assert lattice.describe().startswith(f"ot_matrix_spin_color({Ns},{Nc})")
    if propagator is None:
        value = evenodd(
            numpy.asarray(lattice.mview()[0])
            .view(f"<c{2 * gpt_prec}")
            .reshape(*gpt_latt[::-1], Ns, Ns, Nc, Nc, *gpt_simd[::-1])
            .transpose(8, 0, 9, 1, 10, 2, 11, 3, 4, 5, 6, 7)
            .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc)
            .astype("<c16"),
            [0, 1, 2, 3],
        )
        propagator = LatticePropagator(latt_info, value)
        propagator.toDevice()
        return propagator
    else:
        assert latt_info.size == propagator.latt_info.size
        gpt_shape = [i for sl in zip(gpt_simd[::-1], gpt_latt[::-1]) for i in sl]
        lattice.mview()[0][:] = (
            propagator.lexico()
            .astype(f"<c{2 * gpt_prec}")
            .reshape(*gpt_shape, Ns, Ns, Nc, Nc)
            .transpose(1, 3, 5, 7, 8, 9, 10, 11, 0, 2, 4, 6)
            .copy()  # .view("|u1") requires this
            .view("|u1")
            .reshape(-1)
        )
