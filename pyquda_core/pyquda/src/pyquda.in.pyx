import os
import sys
import io
from contextlib import contextmanager

from mpi4py import MPI

from cython.operator cimport dereference
from libc.stdio cimport stdout
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp
from numpy cimport ndarray
ctypedef double complex double_complex

from pyquda_comm.pointer cimport Pointer, Pointers, _NDArray
cimport quda


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

ctypedef struct MapData:
    int ndim
    int dims[6]

cdef MapData map_data

cdef int defaultMap(const int *coords, void *fdata) noexcept:
    cdef MapData *md = <MapData *>fdata
    cdef int rank = 0
    for i in range(md.ndim):
        rank = rank * md.dims[i] + coords[i]
    return rank

cdef int reversedMap(const int *coords, void *fdata) noexcept:
    cdef MapData *md = <MapData *>fdata
    cdef int rank = coords[md.ndim - 1]
    for i in range(md.ndim - 2, -1, -1):
        rank = rank * md.dims[i] + coords[i]
    return rank

def _defaultRankFromCoord(coords: Sequence[int], dims: Sequence[int]) -> int:
    rank = 0
    for coord, dim in zip(coords, dims):
        rank = rank * dim + coord
    return rank


def _defaultCoordFromRank(rank: int, dims: Sequence[int]) -> List[int]:
    coords = []
    for dim in dims[::-1]:
        coords.append(rank % dim)
        rank = rank // dim
    return coords[::-1]

shared_rank_list: list = None

cdef int sharedMap(const int *coords, void *fdata) noexcept:
    cdef MapData *md = <MapData *>fdata
    global shared_rank_list
    grid_size = [md.dims[i] for i in range(md.ndim)]
    if shared_rank_list is None:
        comm = MPI.COMM_WORLD
        shared_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        shared_size = shared_comm.Get_size()
        shared_rank = shared_comm.Get_rank()
        shared_root = shared_comm.bcast(comm.Get_rank())
        node_rank = comm.allgather(shared_root).index(shared_root)
        node_grid_size = [G for G in grid_size]
        shared_grid_size = [1 for _ in grid_size]
        dim, last_dim = 0, len(grid_size) - 1
        while shared_size > 1:
            for prime in [2, 3, 5]:
                if node_grid_size[dim] % prime == 0 and shared_size % prime == 0:
                    node_grid_size[dim] //= prime
                    shared_grid_size[dim] *= prime
                    shared_size //= prime
                    last_dim = dim
                    break
            else:
                if last_dim == dim:
                    raise ValueError("GlobalSharedMemory::GetShmDims failed")
            dim = (dim + 1) % len(grid_size)
        grid_coord = [
            n * S + s
            for n, S, s in zip(
                _defaultCoordFromRank(node_rank, node_grid_size),
                shared_grid_size,
                _defaultCoordFromRank(shared_rank, shared_grid_size),
            )
        ]
        shared_rank_list = comm.allgather(_defaultRankFromCoord(grid_coord, grid_size))

    cdef int rank = shared_rank_list.index(defaultMap(coords, fdata))
    return rank

def initCommsGridQuda(int nDim, list dims, const char grid_map[]):
    cdef int _dims[4]
    _dims = dims
    map_data.ndim = nDim
    for i in range(nDim):
        map_data.dims[i] = _dims[i]
    if strcmp(grid_map, "default") == 0:
        quda.initCommsGridQuda(nDim, _dims, NULL, NULL)
    elif strcmp(grid_map, "reversed") == 0:
        quda.initCommsGridQuda(nDim, _dims, reversedMap, <void *>(&map_data))
    elif strcmp(grid_map, "shared") == 0:
        quda.initCommsGridQuda(nDim, _dims, sharedMap, <void *>(&map_data))

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

def loadGaugeQuda(h_gauge, QudaGaugeParam param):
    _h_gauge = _NDArray(h_gauge, 2)
    quda.loadGaugeQuda(_h_gauge.ptr, &param.param)

def freeGaugeQuda():
    quda.freeGaugeQuda()

def freeUniqueGaugeQuda(quda.QudaLinkType link_type):
    quda.freeUniqueGaugeQuda(link_type)

def freeGaugeSmearedQuda():
    quda.freeGaugeSmearedQuda()

def saveGaugeQuda(h_gauge, QudaGaugeParam param):
    _h_gauge = _NDArray(h_gauge, 2)
    quda.saveGaugeQuda(_h_gauge.ptr, &param.param)

def loadCloverQuda(h_clover, h_clovinv, QudaInvertParam inv_param):
    _h_clover = _NDArray(h_clover, 1)
    _h_clovinv = _NDArray(h_clovinv, 1)
    quda.loadCloverQuda(_h_clover.ptr, _h_clovinv.ptr, &inv_param.param)

def freeCloverQuda():
    quda.freeCloverQuda()

# QUDA only declares lanczosQuda
# def lanczosQuda(int k0, int m, Pointer hp_Apsi, Pointer hp_r, Pointer hp_V, Pointer hp_alpha, Pointer hp_beta, QudaEigParam eig_param):

def eigensolveQuda(h_evecs, ndarray[double_complex, ndim=1] h_evals, QudaEigParam param):
    _h_evecs = _NDArray(h_evecs, 2)
    _h_evals = _NDArray(h_evals)
    quda.eigensolveQuda(_h_evecs.ptrs, <double_complex *>_h_evals.ptr, &param.param)

def invertQuda(h_x, h_b, QudaInvertParam param):
    _h_x = _NDArray(h_x, 1)
    _h_b = _NDArray(h_b, 1)
    quda.invertQuda(_h_x.ptr, _h_b.ptr, &param.param)

def invertMultiSrcQuda(_hp_x, _hp_b, QudaInvertParam param):
    __hp_x = _NDArray(_hp_x, 2)
    __hp_b = _NDArray(_hp_b, 2)
    quda.invertMultiSrcQuda(__hp_x.ptrs, __hp_b.ptrs, &param.param)

def invertMultiShiftQuda(_hp_x, _hp_b, QudaInvertParam param):
    __hp_x = _NDArray(_hp_x, 2)
    __hp_b = _NDArray(_hp_b, 1)
    quda.invertMultiShiftQuda(__hp_x.ptrs, __hp_b.ptr, &param.param)

def newMultigridQuda(QudaMultigridParam param):
    mg_instance = Pointer("void")
    mg_instance.set_ptr(quda.newMultigridQuda(&param.param))
    return mg_instance

def destroyMultigridQuda(Pointer mg_instance):
    quda.destroyMultigridQuda(mg_instance.ptr)

def updateMultigridQuda(Pointer mg_instance, QudaMultigridParam param):
    quda.updateMultigridQuda(mg_instance.ptr, &param.param)

def dumpMultigridQuda(Pointer mg_instance, QudaMultigridParam param):
    quda.dumpMultigridQuda(mg_instance.ptr, &param.param)

def dslashQuda(h_out, h_in, QudaInvertParam inv_param, quda.QudaParity parity):
    _h_out = _NDArray(h_out, 1)
    _h_in = _NDArray(h_in, 1)
    quda.dslashQuda(_h_out.ptr, _h_in.ptr, &inv_param.param, parity)

def dslashMultiSrcQuda(_hp_x, _hp_b, QudaInvertParam param, quda.QudaParity parity):
    __hp_x = _NDArray(_hp_x, 2)
    __hp_b = _NDArray(_hp_b, 2)
    quda.dslashMultiSrcQuda(__hp_x.ptrs, __hp_b.ptrs, &param.param, parity)

def cloverQuda(h_out, h_in, QudaInvertParam inv_param, quda.QudaParity parity, int inverse):
    _h_out = _NDArray(h_out, 1)
    _h_in = _NDArray(h_in, 1)
    quda.cloverQuda(_h_out.ptr, _h_in.ptr, &inv_param.param, parity, inverse)

def MatQuda(h_out, h_in, QudaInvertParam inv_param):
    _h_out = _NDArray(h_out, 1)
    _h_in = _NDArray(h_in, 1)
    quda.MatQuda(_h_out.ptr, _h_in.ptr, &inv_param.param)

def MatDagMatQuda(h_out, h_in, QudaInvertParam inv_param):
    _h_out = _NDArray(h_out, 1)
    _h_in = _NDArray(h_in, 1)
    quda.MatDagMatQuda(_h_out.ptr, _h_in.ptr, &inv_param.param)

# void set_dim(int *)
# void pack_ghost(void **cpuLink, void **cpuGhost, int nFace, QudaPrecision precision)

def computeKSLinkQuda(fatlink, longlink, ulink, inlink, ndarray[double, ndim=1] path_coeff, QudaGaugeParam param):
    _fatlink = _NDArray(fatlink, 2)
    _longlink = _NDArray(longlink, 2)
    _ulink = _NDArray(ulink, 2)
    _inlink = _NDArray(inlink, 2)
    _path_coeff = _NDArray(path_coeff)
    quda.computeKSLinkQuda(_fatlink.ptr, _longlink.ptr, _ulink.ptr, _inlink.ptr, <double *>_path_coeff.ptr, &param.param)

def computeTwoLinkQuda(twolink, inlink, QudaGaugeParam param):
    _twolink = _NDArray(twolink, 2)
    _inlink = _NDArray(inlink, 2)
    quda.computeTwoLinkQuda(_twolink.ptr, _inlink.ptr, &param.param)

def momResidentQuda(mom, QudaGaugeParam param):
    _mom = _NDArray(mom, 2)
    quda.momResidentQuda(_mom.ptr, &param.param)

def computeGaugeForceQuda(mom, sitelink, ndarray[int, ndim=3] input_path_buf, ndarray[int, ndim=1] path_length, ndarray[double, ndim=1] loop_coeff, int num_paths, int max_length, double dt, QudaGaugeParam qudaGaugeParam):
    _mom = _NDArray(mom, 2)
    _sitelink = _NDArray(sitelink, 2)
    _input_path_buf = _NDArray(input_path_buf)
    _path_length = _NDArray(path_length)
    _loop_coeff = _NDArray(loop_coeff)
    return quda.computeGaugeForceQuda(_mom.ptr, _sitelink.ptr, <int ***>_input_path_buf.ptrss, <int *>_path_length.ptr, <double *>_loop_coeff.ptr, num_paths, max_length, dt, &qudaGaugeParam.param)

def computeGaugePathQuda(out, sitelink, ndarray[int, ndim=3] input_path_buf, ndarray[int, ndim=1] path_length, ndarray[double, ndim=1] loop_coeff, int num_paths, int max_length, double dt, QudaGaugeParam qudaGaugeParam):
    _out = _NDArray(out, 2)
    _sitelink = _NDArray(sitelink, 2)
    _input_path_buf = _NDArray(input_path_buf)
    _path_length = _NDArray(path_length)
    _loop_coeff = _NDArray(loop_coeff)
    return quda.computeGaugePathQuda(_out.ptr, _sitelink.ptr, <int ***>_input_path_buf.ptrss, <int *>_path_length.ptr, <double *>_loop_coeff.ptr, num_paths, max_length, dt, &qudaGaugeParam.param)

def computeGaugeLoopTraceQuda(ndarray[double_complex, ndim=1] traces, ndarray[int, ndim=2] input_path_buf, ndarray[int, ndim=1] path_length, ndarray[double, ndim=1] loop_coeff, int num_paths, int max_length, double factor):
    _traces = _NDArray(traces)
    _input_path_buf = _NDArray(input_path_buf)
    _path_length = _NDArray(path_length)
    _loop_coeff = _NDArray(loop_coeff)
    quda.computeGaugeLoopTraceQuda(<double_complex *>_traces.ptr, <int **>_input_path_buf.ptrs, <int *>_path_length.ptr, <double *>_loop_coeff.ptr, num_paths, max_length, factor)

def updateGaugeFieldQuda(gauge, momentum, double dt, int conj_mom, int exact, QudaGaugeParam param):
    _gauge = _NDArray(gauge, 2)
    _momentum = _NDArray(momentum, 2)
    quda.updateGaugeFieldQuda(_gauge.ptr, _momentum.ptr, dt, conj_mom, exact, &param.param)

def staggeredPhaseQuda(gauge_h, QudaGaugeParam param):
    _gauge_h = _NDArray(gauge_h, 2)
    quda.staggeredPhaseQuda(_gauge_h.ptr, &param.param)

def projectSU3Quda(gauge_h, double tol, QudaGaugeParam param):
    _gauge_h = _NDArray(gauge_h, 2)
    quda.projectSU3Quda(_gauge_h.ptr, tol, &param.param)

def momActionQuda(momentum, QudaGaugeParam param):
    _momentum = _NDArray(momentum, 2)
    return quda.momActionQuda(_momentum.ptr, &param.param)

# void* createGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param)
# void saveGaugeFieldQuda(void* outGauge, void* inGauge, QudaGaugeParam* param)
# void destroyGaugeFieldQuda(void* gauge)

def createCloverQuda(QudaInvertParam param):
    quda.createCloverQuda(&param.param)

def computeCloverForceQuda(mom, double dt, x, ndarray[double, ndim=1] coeff, double kappa2, double ck, int nvector, double multiplicity, QudaGaugeParam gauge_param, QudaInvertParam inv_param):
    _mom = _NDArray(mom, 2)
    _x = _NDArray(x, 2)
    _coeff = _NDArray(coeff)
    quda.computeCloverForceQuda(_mom.ptr, dt, _x.ptrs, NULL, <double *>_coeff.ptr, kappa2, ck, nvector, multiplicity, NULL, &gauge_param.param, &inv_param.param)

# void computeStaggeredForceQuda(void *mom, double dt, double delta, void *gauge, void **x, QudaGaugeParam *gauge_param, QudaInvertParam *invert_param)

def computeHISQForceQuda(momentum, double dt, ndarray[double, ndim=1] level2_coeff, ndarray[double, ndim=1] fat7_coeff, w_link, v_link, u_link, quark, int num, int num_naik, ndarray[double, ndim=2] coeff, QudaGaugeParam param):
    _momentum = _NDArray(momentum, 2)
    _level2_coeff = _NDArray(level2_coeff)
    _fat7_coeff = _NDArray(fat7_coeff)
    _w_link = _NDArray(w_link, 2)
    _v_link = _NDArray(v_link, 2)
    _u_link = _NDArray(u_link, 2)
    _quark = _NDArray(quark, 2)
    _coeff = _NDArray(coeff)
    quda.computeHISQForceQuda(_momentum.ptr, dt, <double *>_level2_coeff.ptr, <double *>_fat7_coeff.ptr, _w_link.ptr, _v_link.ptr, _u_link.ptr, _quark.ptrs, num, num_naik, <double **>_coeff.ptrs, &param.param)

def gaussGaugeQuda(unsigned long long seed, double sigma):
    quda.gaussGaugeQuda(seed, sigma)

def gaussMomQuda(unsigned long long seed, double sigma):
    quda.gaussMomQuda(seed, sigma)

def plaqQuda():
    cdef double[3] plaq
    quda.plaqQuda(plaq)
    return plaq

def polyakovLoopQuda(int dir):
    cdef double[2] ploop
    quda.polyakovLoopQuda(ploop, dir)
    return ploop

# void copyExtendedResidentGaugeQuda(void *resident_gauge)

def performWuppertalnStep(h_out, h_in, QudaInvertParam param, unsigned int n_steps, double alpha):
    _h_out = _NDArray(h_out, 1)
    _h_in = _NDArray(h_in, 1)
    quda.performWuppertalnStep(_h_out.ptr, _h_in.ptr, &param.param, n_steps, alpha)

def performGaugeSmearQuda(QudaGaugeSmearParam smear_param, QudaGaugeObservableParam obs_param):
    quda.performGaugeSmearQuda(&smear_param.param, &obs_param.param)

def performWFlowQuda(QudaGaugeSmearParam smear_param, QudaGaugeObservableParam obs_param):
    quda.performWFlowQuda(&smear_param.param, &obs_param.param)

def gaugeObservablesQuda(QudaGaugeObservableParam param):
    quda.gaugeObservablesQuda(&param.param)

def contractQuda(x, y, result, quda.QudaContractType cType, QudaInvertParam param, ndarray[int, ndim=1] X):
    _x = _NDArray(x, 1)
    _y = _NDArray(y, 1)
    _result = _NDArray(result, 1)
    _X = _NDArray(X)
    quda.contractQuda(_x.ptr, _y.ptr, _result.ptr, cType, &param.param, <int *>_X.ptr)

def computeGaugeFixingOVRQuda(gauge, unsigned int gauge_dir, unsigned int Nsteps, unsigned int verbose_interval, double relax_boost, double tolerance, unsigned int reunit_interval, unsigned int stopWtheta, QudaGaugeParam param):
    _gauge = _NDArray(gauge, 2)
    return quda.computeGaugeFixingOVRQuda(_gauge.ptr, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta, &param.param)

def computeGaugeFixingFFTQuda(gauge, unsigned int gauge_dir, unsigned int Nsteps, unsigned int verbose_interval, double alpha, unsigned int autotune, double tolerance, unsigned int stopWtheta, QudaGaugeParam param):
    _gauge = _NDArray(gauge, 2)
    return quda.computeGaugeFixingFFTQuda(_gauge.ptr, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta, &param.param)

def blasGEMMQuda(arrayA, arrayB, arrayC, quda.QudaBoolean native, QudaBLASParam param):
    _arrayA = _NDArray(arrayA, 1)
    _arrayB = _NDArray(arrayB, 1)
    _arrayC = _NDArray(arrayC, 1)
    quda.blasGEMMQuda(_arrayA.ptr, _arrayB.ptr, _arrayC.ptr, native, &param.param)

def blasLUInvQuda(Ainv, A, quda.QudaBoolean use_native, QudaBLASParam param):
    _Ainv = _NDArray(Ainv, 1)
    _A = _NDArray(A, 1)
    quda.blasLUInvQuda(_Ainv.ptr, _A.ptr, use_native, &param.param)

def flushChronoQuda(int index):
    quda.flushChronoQuda(index)

def newDeflationQuda(QudaEigParam param):
    df_instance = Pointer("void")
    df_instance.set_ptr(quda.newDeflationQuda(&param.param))
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

def performTwoLinkGaussianSmearNStep(h_in, QudaQuarkSmearParam smear_param):
    _h_in = _NDArray(h_in, 1)
    quda.performTwoLinkGaussianSmearNStep(_h_in.ptr, &smear_param.param)
