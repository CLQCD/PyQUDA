import numpy as np

from .core import getRankFromCoord, evenodd, LatticeLink, LatticeGauge, LatticeFermion, X, Y, Z, T


def quasiAxialGaugeFixing(gauge: LatticeGauge, dir: int):
    import cupy
    from opt_einsum import contract

    GLi, Gi, gi, Li = (
        gauge.latt_info.global_size[dir],
        gauge.latt_info.grid_size[dir],
        gauge.latt_info.grid_coord[dir],
        gauge.latt_info.size[dir],
    )
    Nc = gauge.latt_info.Nc
    comm = gauge.latt_info.mpi_comm

    def get_neighbor_rank(value: int):
        return getRankFromCoord(
            [gauge.latt_info.grid_coord[i] if i != dir else value for i in range(len(gauge.latt_info.grid_coord))],
            gauge.latt_info.grid_size,
        )

    axes = [0, 1, 2, 3, 4, 5]
    axes[0], axes[3 - dir] = axes[3 - dir], axes[0]
    gauge_prod = gauge[dir].lexico().transpose(*axes)
    axes_shape = gauge_prod.shape
    gauge_prod = cupy.array(gauge_prod.reshape(Li, -1, Nc, Nc))
    for i in range(1, Li):
        gauge_prod[i] = contract("xab,xbc->xac", gauge_prod[i - 1], gauge_prod[i])

    if Gi > 1:
        buf_shape, buf_dtype = gauge_prod[-1].shape, gauge_prod.dtype
        if gi != 0:
            buf = np.empty(buf_shape, dtype=buf_dtype)
            comm.Recv(buf, get_neighbor_rank((gi - 1) % Gi))
            for i in range(0, Li):
                gauge_prod[i] = contract("xab,xbc->xac", buf, gauge_prod[i])
        if gi != Gi - 1:
            buf = gauge_prod[-1].get()
            comm.Send(buf, get_neighbor_rank((gi + 1) % Gi))
        if gi == Gi - 1:
            buf = gauge_prod[-1].get()
            for i in range(0, Gi - 1):
                comm.Send(buf, get_neighbor_rank(i))
        else:
            buf = np.empty(buf_shape, dtype=buf_dtype)
            comm.Recv(buf, get_neighbor_rank(Gi - 1))
    else:
        buf = gauge_prod[-1].get()

    w, v = np.linalg.eig(buf)
    w, v = cupy.array(w), cupy.array(v)
    w = cupy.angle(w)
    rotate = cupy.zeros_like(gauge_prod)
    rotate[0] = contract("xab,xb->xab", v, cupy.exp(1j * (gi * Li) / GLi * w))
    for i in range(1, Li):
        rotate[i] = contract("xba,xbc,xc->xac", gauge_prod[i - 1].conj(), v, cupy.exp(1j * (i + gi * Li) / GLi * w))
    rotate = LatticeLink(gauge.latt_info, evenodd(rotate.reshape(*axes_shape).transpose(*axes).get(), [0, 1, 2, 3]))
    rotate.toDevice()
    rotate_ = LatticeFermion(gauge.latt_info)
    rotate.pack(rotate_)
    gauge.data = contract("wtzyxba,dwtzyxbc->dwtzyxac", rotate.data.conj(), gauge.data)
    gauge.gauge_dirac.loadGauge(gauge)
    gauge[X].unpack(gauge.gauge_dirac.covDev(rotate_, X))
    gauge[Y].unpack(gauge.gauge_dirac.covDev(rotate_, Y))
    gauge[Z].unpack(gauge.gauge_dirac.covDev(rotate_, Z))
    gauge[T].unpack(gauge.gauge_dirac.covDev(rotate_, T))
    return rotate
