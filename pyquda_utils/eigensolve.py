from pyquda import pyquda as quda
from pyquda.enum_quda import QudaDslashType, QudaBoolean, QudaEigType, QudaEigSpectrumType
from pyquda.field import LatticeGauge, MultiLatticeStaggeredFermion


def laplace3d(
    gauge: LatticeGauge, n_ev: int, n_kr: int, tol: int, max_restarts: int, poly_deg: int = 1, poly_cut: float = 0.0
):
    import numpy

    latt_info = gauge.latt_info
    gauge.gauge_dirac.invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
    gauge.gauge_dirac.invert_param.mass = -1
    gauge.gauge_dirac.invert_param.kappa = 1 / 6
    gauge.gauge_dirac.invert_param.laplace3D = 3
    gauge.gauge_dirac.loadGauge(gauge)

    eig_param = quda.QudaEigParam()
    eig_param.invert_param = gauge.gauge_dirac.invert_param
    eig_param.eig_type = QudaEigType.QUDA_EIG_TR_LANCZOS_3D
    eig_param.use_dagger = QudaBoolean.QUDA_BOOLEAN_FALSE
    eig_param.use_norm_op = QudaBoolean.QUDA_BOOLEAN_FALSE
    eig_param.use_pc = QudaBoolean.QUDA_BOOLEAN_FALSE
    eig_param.compute_gamma5 = QudaBoolean.QUDA_BOOLEAN_FALSE
    eig_param.spectrum = QudaEigSpectrumType.QUDA_SPECTRUM_SR_EIG
    eig_param.n_ev = n_ev
    eig_param.n_kr = n_kr
    eig_param.n_conv = n_ev
    eig_param.tol = tol
    eig_param.ortho_dim = 3
    eig_param.ortho_dim_size_local = latt_info.Lt
    eig_param.vec_infile = b""
    eig_param.vec_outfile = b""
    eig_param.max_restarts = max_restarts
    eig_param.use_poly_acc = QudaBoolean(poly_deg > 1)
    eig_param.poly_deg = poly_deg
    eig_param.a_min = poly_cut
    eig_param.a_max = 2

    evals = numpy.empty((latt_info.GLt, n_ev), "<c16")
    evecs = MultiLatticeStaggeredFermion(latt_info, n_ev)
    quda.eigensolveQuda(evecs.data_ptrs, evals.reshape(-1), eig_param)
    return evals.real, evecs
