import os
import sys
import io
from contextlib import contextmanager

import cython

from libc.stdio cimport stdout

cimport quda
ctypedef double complex double_complex
from pyquda.pointer cimport Pointer, Pointers, Pointerss


@contextmanager
def redirect_stdout(value: bytearray):
    stdout_fd = sys.stdout.fileno()
    stdout_dup_fd = os.dup(stdout_fd)
    pipe_out, pipe_in = os.pipe()
    os.dup2(pipe_in, stdout_fd)

    yield

    sys.stdout.write(b"\x00".decode(sys.stdout.encoding))
    sys.stdout.flush()
    os.close(pipe_in)
    with io.FileIO(pipe_out, closefd=True) as fio:
        buffer = fio.read(4096)
        while b"\00" not in buffer:
            value.extend(buffer)
            buffer = fio.read(4096)
        value.extend(buffer)
    os.dup2(stdout_dup_fd, stdout_fd)
    os.close(stdout_dup_fd)

cdef class QudaGaugeParam:
    cdef quda.QudaGaugeParam param

    def __init__(self):
        self.param = quda.newQudaGaugeParam()

    def __repr__(self):
        value = bytearray()
        with redirect_stdout(value):
            quda.printQudaGaugeParam(&self.param)
        return value.decode(sys.stdout.encoding)

    cdef from_ptr(self, quda.QudaGaugeParam *ptr):
        self.param = cython.operator.dereference(ptr)

##%%!! QudaGaugeParam

cdef class QudaInvertParam:
    cdef quda.QudaInvertParam param

    def __init__(self):
        self.param = quda.newQudaInvertParam()

    def __repr__(self):
        value = bytearray()
        with redirect_stdout(value):
            quda.printQudaInvertParam(&self.param)
        return value.decode(sys.stdout.encoding)

    cdef from_ptr(self, quda.QudaInvertParam *ptr):
        self.param = cython.operator.dereference(ptr)

##%%!! QudaInvertParam

cdef class QudaMultigridParam:
    cdef quda.QudaMultigridParam param

    def __init__(self):
        self.param = quda.newQudaMultigridParam()

    def __repr__(self):
        value = bytearray()
        with redirect_stdout(value):
            quda.printQudaMultigridParam(&self.param)
        return value.decode(sys.stdout.encoding)

    cdef from_ptr(self, quda.QudaMultigridParam *ptr):
        self.param = cython.operator.dereference(ptr)

##%%!! QudaMultigridParam

cdef class QudaEigParam:
    cdef quda.QudaEigParam param

    def __init__(self):
        self.param = quda.newQudaEigParam()

    def __repr__(self):
        value = bytearray()
        with redirect_stdout(value):
            quda.printQudaEigParam(&self.param)
        return value.decode(sys.stdout.encoding)

    cdef from_ptr(self, quda.QudaEigParam *ptr):
        self.param = cython.operator.dereference(ptr)

##%%!! QudaEigParam

cdef class QudaGaugeObservableParam:
    cdef quda.QudaGaugeObservableParam param

    def __init__(self):
        self.param = quda.newQudaGaugeObservableParam()

    def __repr__(self):
        value = bytearray()
        with redirect_stdout(value):
            quda.printQudaGaugeObservableParam(&self.param)
        return value.decode(sys.stdout.encoding)

    cdef from_ptr(self, quda.QudaGaugeObservableParam *ptr):
        self.param = cython.operator.dereference(ptr)

##%%!! QudaGaugeObservableParam

cdef class QudaGaugeSmearParam:
    cdef quda.QudaGaugeSmearParam param

    def __init__(self):
        self.param = quda.newQudaGaugeSmearParam()

    # def __repr__(self):
    #     value = bytearray()
    #     with redirect_stdout(value):
    #         quda.printQudaGaugeSmearParam(&self.param)
    #     return value.decode(sys.stdout.encoding)

    cdef from_ptr(self, quda.QudaGaugeSmearParam *ptr):
        self.param = cython.operator.dereference(ptr)

##%%!! QudaGaugeSmearParam

cdef class QudaBLASParam:
    cdef quda.QudaBLASParam param

    def __init__(self):
        self.param = quda.newQudaBLASParam()

    def __repr__(self):
        value = bytearray()
        with redirect_stdout(value):
            quda.printQudaBLASParam(&self.param)
        return value.decode(sys.stdout.encoding)

    cdef from_ptr(self, quda.QudaBLASParam *ptr):
        self.param = cython.operator.dereference(ptr)

##%%!! QudaBLASParam

def setVerbosityQuda(quda.QudaVerbosity verbosity, const char prefix[]):
    quda.setVerbosityQuda(verbosity, prefix, stdout)

def initCommsGridQuda(int nDim, list dims):
    assert nDim == 4 and len(dims) >= 4
    cdef int c_dims[4]
    c_dims = dims
    quda.initCommsGridQuda(nDim, c_dims, NULL, NULL)

def initQudaDevice(int device):
    quda.initQudaDevice(device)

def initQudaMemory():
    quda.initQudaMemory()

def initQuda(int device):
    quda.initQuda(device)

def endQuda():
    quda.endQuda()

def updateR():
    quda.updateR()

def loadGaugeQuda(Pointers h_gauge, QudaGaugeParam param):
    assert h_gauge.dtype == "void"
    quda.loadGaugeQuda(h_gauge.ptr, &param.param)

def freeGaugeQuda():
    quda.freeGaugeQuda()

def freeUniqueGaugeQuda(quda.QudaLinkType link_type):
    quda.freeUniqueGaugeQuda(link_type)

def freeGaugeSmearedQuda():
    quda.freeGaugeSmearedQuda()

def saveGaugeQuda(Pointers h_gauge, QudaGaugeParam param):
    assert h_gauge.dtype == "void"
    quda.saveGaugeQuda(h_gauge.ptr, &param.param)

def loadCloverQuda(Pointer h_clover, Pointer h_clovinv, QudaInvertParam inv_param):
    assert h_clover.dtype == "void"
    assert h_clovinv.dtype == "void"
    quda.loadCloverQuda(h_clover.ptr, h_clovinv.ptr, &inv_param.param)

def freeCloverQuda():
    quda.freeCloverQuda()

# def lanczosQuda(int k0, int m, Pointer hp_Apsi, Pointer hp_r, Pointer hp_V, Pointer hp_alpha, Pointer hp_beta, QudaEigParam eig_param)
# def eigensolveQuda(Pointers h_evecs, Pointer<double_complex> h_evals, QudaEigParam param)

def invertQuda(Pointer h_x, Pointer h_b, QudaInvertParam param):
    assert h_x.dtype == "void"
    assert h_b.dtype == "void"
    quda.invertQuda(h_x.ptr, h_b.ptr, &param.param)

# def invertMultiSrcQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, Pointer h_gauge, QudaGaugeParam gauge_param)
# def invertMultiSrcStaggeredQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, Pointer milc_fatlinks, Pointer milc_longlinks, QudaGaugeParam gauge_param)
# def invertMultiSrcCloverQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, Pointer h_gauge, QudaGaugeParam gauge_param, Pointer h_clover, Pointer h_clovinv)

def invertMultiShiftQuda(Pointers _hp_x, Pointer _hp_b, QudaInvertParam param):
    assert _hp_x.dtype == "void"
    assert _hp_b.dtype == "void"
    quda.invertMultiShiftQuda(_hp_x.ptrs, _hp_b.ptr, &param.param)

def newMultigridQuda(QudaMultigridParam param) -> Pointer:
    mg_instance = Pointer("void")
    cdef void *ptr = quda.newMultigridQuda(&param.param)
    mg_instance.set_ptr(ptr)
    return mg_instance

def destroyMultigridQuda(Pointer mg_instance):
    quda.destroyMultigridQuda(mg_instance.ptr)

def updateMultigridQuda(Pointer mg_instance, QudaMultigridParam param):
    quda.updateMultigridQuda(mg_instance.ptr, &param.param)

def dumpMultigridQuda(Pointer mg_instance, QudaMultigridParam param):
    quda.dumpMultigridQuda(mg_instance.ptr, &param.param)

def dslashQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param, quda.QudaParity parity):
    assert h_out.dtype == "void"
    assert h_in.dtype == "void"
    quda.dslashQuda(h_out.ptr, h_in.ptr, &inv_param.param, parity)

# def dslashMultiSrcQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, QudaParity parity, Pointer h_gauge, QudaGaugeParam gauge_param)
# def dslashMultiSrcStaggeredQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, QudaParity parity, Pointers milc_fatlinks, Pointers milc_longlinks, QudaGaugeParam gauge_param)
# def dslashMultiSrcCloverQuda(Pointers_hp_x, Pointers_hp_b, QudaInvertParam param, QudaParity parity, Pointer h_gauge, QudaGaugeParam gauge_param, Pointer h_clover, Pointer h_clovinv)

def cloverQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param, quda.QudaParity parity, int inverse):
    assert h_out.dtype == "void"
    assert h_in.dtype == "void"
    quda.cloverQuda(h_out.ptr, h_in.ptr, &inv_param.param, parity, inverse)

def MatQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param):
    assert h_out.dtype == "void"
    assert h_in.dtype == "void"
    quda.MatQuda(h_out.ptr, h_in.ptr, &inv_param.param)

def MatDagMatQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param):
    assert h_out.dtype == "void"
    assert h_in.dtype == "void"
    quda.MatDagMatQuda(h_out.ptr, h_in.ptr, &inv_param.param)

# void set_dim(int *)
# void pack_ghost(void **cpuLink, void **cpuGhost, int nFace, QudaPrecision precision)

def computeKSLinkQuda(Pointers fatlink, Pointers longlink, Pointers ulink, Pointers inlink, Pointer path_coeff, QudaGaugeParam param):
    assert fatlink.dtype == "void"
    assert longlink.dtype == "void"
    assert ulink.dtype == "void"
    assert inlink.dtype == "void"
    assert path_coeff.dtype == "double"
    quda.computeKSLinkQuda(fatlink.ptr, longlink.ptr, ulink.ptr, inlink.ptr, <double *>path_coeff.ptr, &param.param)

def computeTwoLinkQuda(Pointers twolink, Pointers inlink, QudaGaugeParam param):
    assert twolink.dtype == "void"
    assert inlink.dtype == "void"
    quda.computeTwoLinkQuda(twolink.ptr, inlink.ptr, &param.param)

def momResidentQuda(Pointers mom, QudaGaugeParam param):
    assert mom.dtype == "void"
    quda.momResidentQuda(mom.ptr, &param.param)

def computeGaugeForceQuda(Pointers mom, Pointers sitelink, Pointerss input_path_buf, Pointer path_length, Pointer loop_coeff, int num_paths, int max_length, double dt, QudaGaugeParam qudaGaugeParam):
    assert mom.dtype == "void"
    assert sitelink.dtype == "void"
    assert input_path_buf.dtype == "int"
    assert path_length.dtype == "int"
    assert loop_coeff.dtype == "double"
    return quda.computeGaugeForceQuda(mom.ptr, sitelink.ptr, <int ***>input_path_buf.ptr, <int *>path_length.ptr, <double *>loop_coeff.ptr, num_paths, max_length, dt, &qudaGaugeParam.param)

def computeGaugePathQuda(Pointers out, Pointers sitelink, Pointerss input_path_buf, Pointer path_length, Pointer loop_coeff, int num_paths, int max_length, double dt, QudaGaugeParam qudaGaugeParam):
    assert out.dtype == "void"
    assert sitelink.dtype == "void"
    assert input_path_buf.dtype == "int"
    assert path_length.dtype == "int"
    assert loop_coeff.dtype == "double"
    return quda.computeGaugePathQuda(out.ptr, sitelink.ptr, <int ***>input_path_buf.ptr, <int *>path_length.ptr, <double *>loop_coeff.ptr, num_paths, max_length, dt, &qudaGaugeParam.param)

def computeGaugeLoopTraceQuda(Pointer traces, Pointers input_path_buf, Pointer path_length, Pointer loop_coeff, int num_paths, int max_length, double factor):
    assert traces.dtype == "double_complex"
    assert input_path_buf.dtype == "int"
    assert path_length.dtype == "int"
    assert loop_coeff.dtype == "double"
    quda.computeGaugeLoopTraceQuda(<double_complex *>traces.ptr, <int **>input_path_buf.ptr, <int *>path_length.ptr, <double *>loop_coeff.ptr, num_paths, max_length, factor)

def updateGaugeFieldQuda(Pointers gauge, Pointers momentum, double dt, int conj_mom, int exact, QudaGaugeParam param):
    assert gauge.dtype == "void"
    assert momentum.dtype == "void"
    quda.updateGaugeFieldQuda(gauge.ptr, momentum.ptr, dt, conj_mom, exact, &param.param)

def staggeredPhaseQuda(Pointers gauge_h, QudaGaugeParam param):
    assert gauge_h.dtype == "void"
    quda.staggeredPhaseQuda(gauge_h.ptr, &param.param)

def projectSU3Quda(Pointers gauge_h, double tol, QudaGaugeParam param):
    assert gauge_h.dtype == "void"
    quda.projectSU3Quda(gauge_h.ptr, tol, &param.param)

def momActionQuda(Pointers momentum, QudaGaugeParam param):
    assert momentum.dtype == "void"
    return quda.momActionQuda(momentum.ptr, &param.param)

# void* createGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param)
# void saveGaugeFieldQuda(void* outGauge, void* inGauge, QudaGaugeParam* param)
# void destroyGaugeFieldQuda(void* gauge)

def createCloverQuda(QudaInvertParam param):
    quda.createCloverQuda(&param.param)

def computeCloverForceQuda(Pointers mom, double dt, Pointers x, Pointer coeff, double kappa2, double ck, int nvector, double multiplicity, QudaGaugeParam gauge_param, QudaInvertParam inv_param):
    assert mom.dtype == "void"
    assert x.dtype == "void"
    assert coeff.dtype == "double"
    quda.computeCloverForceQuda(mom.ptr, dt, x.ptrs, NULL, <double *>coeff.ptr, kappa2, ck, nvector, multiplicity, NULL, &gauge_param.param, &inv_param.param)

# void computeStaggeredForceQuda(void *mom, double dt, double delta, void *gauge, void **x, QudaGaugeParam *gauge_param, QudaInvertParam *invert_param)
# void computeHISQForceQuda(void* momentum, double dt, const double level2_coeff[6], const double fat7_coeff[6], const void* const w_link, const void* const v_link, const void* const u_link, void** quark, int num, int num_naik, double** coeff, QudaGaugeParam* param)

def gaussGaugeQuda(unsigned long long seed, double sigma):
    quda.gaussGaugeQuda(seed, sigma)

def gaussMomQuda(unsigned long long seed, double sigma):
    quda.gaussMomQuda(seed, sigma)

def plaqQuda() -> list:
    cdef double plaq[3]
    quda.plaqQuda(plaq)
    return plaq

def polyakovLoopQuda(int dir) -> list:
    cdef double ploop[2]
    quda.polyakovLoopQuda(ploop, dir)
    return ploop

# void copyExtendedResidentGaugeQuda(void *resident_gauge)

def performWuppertalnStep(Pointer h_out, Pointer h_in, QudaInvertParam param, unsigned int n_steps, double alpha):
    assert h_out.dtype == "void"
    assert h_in.dtype == "void"
    quda.performWuppertalnStep(h_out.ptr, h_in.ptr, &param.param, n_steps, alpha)

def performGaugeSmearQuda(QudaGaugeSmearParam smear_param, QudaGaugeObservableParam obs_param):
    quda.performGaugeSmearQuda(&smear_param.param, &obs_param.param)

def performWFlowQuda(QudaGaugeSmearParam smear_param, QudaGaugeObservableParam obs_param):
    quda.performWFlowQuda(&smear_param.param, &obs_param.param)

def gaugeObservablesQuda(QudaGaugeObservableParam param):
    quda.gaugeObservablesQuda(&param.param)

def contractQuda(Pointer x, Pointer y, Pointer result, quda.QudaContractType cType, QudaInvertParam param, Pointer X):
    assert x.dtype == "void"
    assert y.dtype == "void"
    assert result.dtype == "void"
    assert X.dtype == "int"
    quda.contractQuda(x.ptr, y.ptr, result.ptr, cType, &param.param, <int *>X.ptr)

def computeGaugeFixingOVRQuda(Pointers gauge, unsigned int gauge_dir, unsigned int Nsteps, unsigned int verbose_interval, double relax_boost, double tolerance, unsigned int reunit_interval, unsigned int stopWtheta, QudaGaugeParam param):
    assert gauge.dtype == "void"
    return quda.computeGaugeFixingOVRQuda(gauge.ptr, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta, &param.param)

def computeGaugeFixingFFTQuda(Pointers gauge, unsigned int gauge_dir, unsigned int Nsteps, unsigned int verbose_interval, double alpha, unsigned int autotune, double tolerance, unsigned int stopWtheta, QudaGaugeParam param):
    assert gauge.dtype == "void"
    return quda.computeGaugeFixingFFTQuda(gauge.ptr, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta, &param.param)

def blasGEMMQuda(Pointer arrayA, Pointer arrayB, Pointer arrayC, quda.QudaBoolean native, QudaBLASParam param):
    assert arrayA.dtype == "void"
    assert arrayB.dtype == "void"
    assert arrayC.dtype == "void"
    quda.blasGEMMQuda(arrayA.ptr, arrayB.ptr, arrayC.ptr, native, &param.param)

def blasLUInvQuda(Pointer Ainv, Pointer A, quda.QudaBoolean use_native, QudaBLASParam param):
    assert Ainv.dtype == "void"
    assert A.dtype == "void"
    quda.blasLUInvQuda(Ainv.ptr, A.ptr, use_native, &param.param)

def flushChronoQuda(int index):
    quda.flushChronoQuda(index)

def newDeflationQuda(QudaEigParam param) -> Pointer:
    df_instance = Pointer("void")
    cdef void *ptr = quda.newDeflationQuda(&param.param)
    df_instance.set_ptr(ptr)
    return df_instance

def destroyDeflationQuda(Pointer df_instance):
    assert df_instance.dtype == "void"
    quda.destroyDeflationQuda(df_instance.ptr)

cdef class QudaQuarkSmearParam:
    cdef quda.QudaQuarkSmearParam param

    def __init__(self):
        # self.param = quda.QudaQuarkSmearParam()
        pass

    # def __repr__(self):
    #     value = bytearray()
    #     with redirect_stdout(value):
    #         quda.printQudaQuarkSmearParam(&self.param)
    #     return value.decode(sys.stdout.encoding)

    cdef from_ptr(self, quda.QudaQuarkSmearParam *ptr):
        self.param = cython.operator.dereference(ptr)

##%%!! QudaQuarkSmearParam

def performTwoLinkGaussianSmearNStep(Pointer h_in, QudaQuarkSmearParam smear_param):
    assert h_in.dtype == "void"
    quda.performTwoLinkGaussianSmearNStep(h_in.ptr, &smear_param.param)
