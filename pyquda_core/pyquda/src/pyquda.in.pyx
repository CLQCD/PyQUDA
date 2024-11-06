import os
import sys
import io
from contextlib import contextmanager

from cython.operator cimport dereference
from libc.stdio cimport stdout
from libc.stdlib cimport malloc, free
from numpy cimport ndarray
ctypedef double complex double_complex

cimport quda
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


cdef class _NDArray:
    cdef int n0, n1
    cdef void *ptr
    cdef void **ptrs
    cdef void ***ptrss

    def __cinit__(self, data):
        shape = data.shape
        ndim = data.ndim
        cdef size_t ptr_uint64
        if ndim == 1:
            assert data.flags["C_CONTIGUOUS"]
            self.n0, self.n1 = 0, 0
            ptr_uint64 = data.ctypes.data
            self.ptr = <void *>ptr_uint64
        elif ndim == 2:
            self.n0, self.n1 = shape[0], 0
            self.ptrs = <void **>malloc(shape[0] * sizeof(void *))
            for i in range(shape[0]):
                assert data[i].flags["C_CONTIGUOUS"]
                ptr_uint64 = data[i].ctypes.data
                self.ptrs[i] = <void *>ptr_uint64
        elif ndim == 3:
            self.n0, self.n1 = shape[0], shape[1]
            self.ptrss = <void ***>malloc(shape[0] * sizeof(void **))
            for i in range(shape[0]):
                self.ptrss[i] = <void **>malloc(shape[1] * sizeof(void *))
                for j in range(shape[1]):
                    assert data[i, j].flags["C_CONTIGUOUS"]
                    ptr_uint64 = data[i, j].ctypes.data
                    self.ptrss[i][j] = <void *>ptr_uint64
        else:
            raise NotImplementedError("ndarray.ndim > 3 not implemented yet")

    def __dealloc__(self):
        if self.ptrs:
            free(self.ptrs)
        if self.ptrss:
            for i in range(self.n0):
                free(self.ptrss[i])
            free(self.ptrss)


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
        self.param = dereference(ptr)

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
        self.param = dereference(ptr)

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
        self.param = dereference(ptr)

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
        self.param = dereference(ptr)

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
        self.param = dereference(ptr)

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
        self.param = dereference(ptr)

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
        self.param = dereference(ptr)

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
    quda.loadGaugeQuda(h_gauge.ptr, &param.param)

def freeGaugeQuda():
    quda.freeGaugeQuda()

def freeUniqueGaugeQuda(quda.QudaLinkType link_type):
    quda.freeUniqueGaugeQuda(link_type)

def freeGaugeSmearedQuda():
    quda.freeGaugeSmearedQuda()

def saveGaugeQuda(Pointers h_gauge, QudaGaugeParam param):
    quda.saveGaugeQuda(h_gauge.ptr, &param.param)

def loadCloverQuda(Pointer h_clover, Pointer h_clovinv, QudaInvertParam inv_param):
    quda.loadCloverQuda(h_clover.ptr, h_clovinv.ptr, &inv_param.param)

def freeCloverQuda():
    quda.freeCloverQuda()

# QUDA only declares lanczosQuda
# def lanczosQuda(int k0, int m, Pointer hp_Apsi, Pointer hp_r, Pointer hp_V, Pointer hp_alpha, Pointer hp_beta, QudaEigParam eig_param):

def eigensolveQuda(Pointers h_evecs, ndarray[double_complex, ndim=1] h_evals, QudaEigParam param):
    _h_evals = _NDArray(h_evals)
    quda.eigensolveQuda(h_evecs.ptrs, <double_complex *>_h_evals.ptr, &param.param)

def invertQuda(Pointer h_x, Pointer h_b, QudaInvertParam param):
    quda.invertQuda(h_x.ptr, h_b.ptr, &param.param)

def invertMultiSrcQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param):
    quda.invertMultiSrcQuda(_hp_x.ptrs, _hp_b.ptrs, &param.param)

def invertMultiShiftQuda(Pointers _hp_x, Pointer _hp_b, QudaInvertParam param):
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
    quda.dslashQuda(h_out.ptr, h_in.ptr, &inv_param.param, parity)

def dslashMultiSrcQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, quda.QudaParity parity):
    quda.dslashMultiSrcQuda(_hp_x.ptrs, _hp_b.ptrs, &param.param, parity)

def cloverQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param, quda.QudaParity parity, int inverse):
    quda.cloverQuda(h_out.ptr, h_in.ptr, &inv_param.param, parity, inverse)

def MatQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param):
    quda.MatQuda(h_out.ptr, h_in.ptr, &inv_param.param)

def MatDagMatQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param):
    quda.MatDagMatQuda(h_out.ptr, h_in.ptr, &inv_param.param)

# void set_dim(int *)
# void pack_ghost(void **cpuLink, void **cpuGhost, int nFace, QudaPrecision precision)

def computeKSLinkQuda(Pointers fatlink, Pointers longlink, Pointers ulink, Pointers inlink, ndarray[double, ndim=1] path_coeff, QudaGaugeParam param):
    _path_coeff = _NDArray(path_coeff)
    quda.computeKSLinkQuda(fatlink.ptr, longlink.ptr, ulink.ptr, inlink.ptr, <double *>_path_coeff.ptr, &param.param)

def computeTwoLinkQuda(Pointers twolink, Pointers inlink, QudaGaugeParam param):
    quda.computeTwoLinkQuda(twolink.ptr, inlink.ptr, &param.param)

def momResidentQuda(Pointers mom, QudaGaugeParam param):
    quda.momResidentQuda(mom.ptr, &param.param)

def computeGaugeForceQuda(Pointers mom, Pointers sitelink, ndarray[int, ndim=3] input_path_buf, ndarray[int, ndim=1] path_length, ndarray[double, ndim=1] loop_coeff, int num_paths, int max_length, double dt, QudaGaugeParam qudaGaugeParam):
    _input_path_buf = _NDArray(input_path_buf)
    _path_length = _NDArray(path_length)
    _loop_coeff = _NDArray(loop_coeff)
    return quda.computeGaugeForceQuda(mom.ptr, sitelink.ptr, <int ***>_input_path_buf.ptrss, <int *>_path_length.ptr, <double *>_loop_coeff.ptr, num_paths, max_length, dt, &qudaGaugeParam.param)

def computeGaugePathQuda(Pointers out, Pointers sitelink, ndarray[int, ndim=3] input_path_buf, ndarray[int, ndim=1] path_length, ndarray[double, ndim=1] loop_coeff, int num_paths, int max_length, double dt, QudaGaugeParam qudaGaugeParam):
    _input_path_buf = _NDArray(input_path_buf)
    _path_length = _NDArray(path_length)
    _loop_coeff = _NDArray(loop_coeff)
    return quda.computeGaugePathQuda(out.ptr, sitelink.ptr, <int ***>_input_path_buf.ptrss, <int *>_path_length.ptr, <double *>_loop_coeff.ptr, num_paths, max_length, dt, &qudaGaugeParam.param)

def computeGaugeLoopTraceQuda(ndarray[double_complex, ndim=1] traces, ndarray[int, ndim=2] input_path_buf, ndarray[int, ndim=1] path_length, ndarray[double, ndim=1] loop_coeff, int num_paths, int max_length, double factor):
    _traces = _NDArray(traces)
    _input_path_buf = _NDArray(input_path_buf)
    _path_length = _NDArray(path_length)
    _loop_coeff = _NDArray(loop_coeff)
    quda.computeGaugeLoopTraceQuda(<double_complex *>_traces.ptr, <int **>_input_path_buf.ptrs, <int *>_path_length.ptr, <double *>_loop_coeff.ptr, num_paths, max_length, factor)

def updateGaugeFieldQuda(Pointers gauge, Pointers momentum, double dt, int conj_mom, int exact, QudaGaugeParam param):
    quda.updateGaugeFieldQuda(gauge.ptr, momentum.ptr, dt, conj_mom, exact, &param.param)

def staggeredPhaseQuda(Pointers gauge_h, QudaGaugeParam param):
    quda.staggeredPhaseQuda(gauge_h.ptr, &param.param)

def projectSU3Quda(Pointers gauge_h, double tol, QudaGaugeParam param):
    quda.projectSU3Quda(gauge_h.ptr, tol, &param.param)

def momActionQuda(Pointers momentum, QudaGaugeParam param):
    return quda.momActionQuda(momentum.ptr, &param.param)

# void* createGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param)
# void saveGaugeFieldQuda(void* outGauge, void* inGauge, QudaGaugeParam* param)
# void destroyGaugeFieldQuda(void* gauge)

def createCloverQuda(QudaInvertParam param):
    quda.createCloverQuda(&param.param)

def computeCloverForceQuda(Pointers mom, double dt, Pointers x, ndarray[double, ndim=1] coeff, double kappa2, double ck, int nvector, double multiplicity, QudaGaugeParam gauge_param, QudaInvertParam inv_param):
    _coeff = _NDArray(coeff)
    quda.computeCloverForceQuda(mom.ptr, dt, x.ptrs, NULL, <double *>_coeff.ptr, kappa2, ck, nvector, multiplicity, NULL, &gauge_param.param, &inv_param.param)

# void computeStaggeredForceQuda(void *mom, double dt, double delta, void *gauge, void **x, QudaGaugeParam *gauge_param, QudaInvertParam *invert_param)

def computeHISQForceQuda(Pointers momentum, double dt, ndarray[double, ndim=1] level2_coeff, ndarray[double, ndim=1] fat7_coeff, Pointers w_link, Pointers v_link, Pointer u_link, Pointers quark, int num, int num_naik, ndarray[double, ndim=2] coeff, QudaGaugeParam param):
    _level2_coeff = _NDArray(level2_coeff)
    _fat7_coeff = _NDArray(fat7_coeff)
    _coeff = _NDArray(coeff)
    quda.computeHISQForceQuda(momentum.ptr, dt, <double *>_level2_coeff.ptr, <double *>_fat7_coeff.ptr, w_link.ptr, v_link.ptr, u_link.ptr, quark.ptrs, num, num_naik, <double **>_coeff.ptrs, &param.param)

def gaussGaugeQuda(unsigned long long seed, double sigma):
    quda.gaussGaugeQuda(seed, sigma)

def gaussMomQuda(unsigned long long seed, double sigma):
    quda.gaussMomQuda(seed, sigma)

def plaqQuda() -> list:
    cdef double[3] plaq
    quda.plaqQuda(plaq)
    return plaq

def polyakovLoopQuda(int dir) -> list:
    cdef double[2] ploop
    quda.polyakovLoopQuda(ploop, dir)
    return ploop

# void copyExtendedResidentGaugeQuda(void *resident_gauge)

def performWuppertalnStep(Pointer h_out, Pointer h_in, QudaInvertParam param, unsigned int n_steps, double alpha):
    quda.performWuppertalnStep(h_out.ptr, h_in.ptr, &param.param, n_steps, alpha)

def performGaugeSmearQuda(QudaGaugeSmearParam smear_param, QudaGaugeObservableParam obs_param):
    quda.performGaugeSmearQuda(&smear_param.param, &obs_param.param)

def performWFlowQuda(QudaGaugeSmearParam smear_param, QudaGaugeObservableParam obs_param):
    quda.performWFlowQuda(&smear_param.param, &obs_param.param)

def gaugeObservablesQuda(QudaGaugeObservableParam param):
    quda.gaugeObservablesQuda(&param.param)

def contractQuda(Pointer x, Pointer y, Pointer result, quda.QudaContractType cType, QudaInvertParam param, ndarray[int, ndim=1] X):
    _X = _NDArray(X)
    quda.contractQuda(x.ptr, y.ptr, result.ptr, cType, &param.param, <int *>_X.ptr)

def computeGaugeFixingOVRQuda(Pointers gauge, unsigned int gauge_dir, unsigned int Nsteps, unsigned int verbose_interval, double relax_boost, double tolerance, unsigned int reunit_interval, unsigned int stopWtheta, QudaGaugeParam param):
    return quda.computeGaugeFixingOVRQuda(gauge.ptr, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta, &param.param)

def computeGaugeFixingFFTQuda(Pointers gauge, unsigned int gauge_dir, unsigned int Nsteps, unsigned int verbose_interval, double alpha, unsigned int autotune, double tolerance, unsigned int stopWtheta, QudaGaugeParam param):
    return quda.computeGaugeFixingFFTQuda(gauge.ptr, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta, &param.param)

def blasGEMMQuda(Pointer arrayA, Pointer arrayB, Pointer arrayC, quda.QudaBoolean native, QudaBLASParam param):
    quda.blasGEMMQuda(arrayA.ptr, arrayB.ptr, arrayC.ptr, native, &param.param)

def blasLUInvQuda(Pointer Ainv, Pointer A, quda.QudaBoolean use_native, QudaBLASParam param):
    quda.blasLUInvQuda(Ainv.ptr, A.ptr, use_native, &param.param)

def flushChronoQuda(int index):
    quda.flushChronoQuda(index)

def newDeflationQuda(QudaEigParam param) -> Pointer:
    df_instance = Pointer("void")
    cdef void *ptr = quda.newDeflationQuda(&param.param)
    df_instance.set_ptr(ptr)
    return df_instance

def destroyDeflationQuda(Pointer df_instance):
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
        self.param = dereference(ptr)

##%%!! QudaQuarkSmearParam

def performTwoLinkGaussianSmearNStep(Pointer h_in, QudaQuarkSmearParam smear_param):
    quda.performTwoLinkGaussianSmearNStep(h_in.ptr, &smear_param.param)
