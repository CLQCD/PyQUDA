#pragma once

#
# @file  quda.h
# @brief Main header file for the QUDA library
#
# Note to QUDA developers: When adding new members to QudaGaugeParam
# and QudaInvertParam, be sure to update lib/check_params.h as well
# as the Fortran interface in lib/quda_fortran.F90.
#

from libcpp cimport bool
from enum_quda cimport *
from libc.stdio cimport FILE
# from quda_define cimport *
from quda_constants cimport *

cdef extern from "complex.h":
    pass
ctypedef double complex double_complex

cdef extern from "quda.h":

    #
    # Parameters having to do with the gauge field or the
    # interpretation of the gauge field by various Dirac operators
    #
    ctypedef struct QudaGaugeParam:
        pass

    #
    # Parameters relating to the solver and the choice of Dirac operator.
    #
    ctypedef struct QudaInvertParam:
        pass

    # Parameter set for solving eigenvalue problems.
    ctypedef struct QudaEigParam:
        pass

    ctypedef struct QudaMultigridParam:
        pass

    ctypedef struct QudaGaugeObservableParam:
        pass

    ctypedef struct QudaGaugeSmearParam:
        pass

    ctypedef struct QudaBLASParam:
        pass

    #
    # Interface functions, found in interface_quda.cpp
    #

    #
    # Set parameters related to status reporting.
    #
    # In typical usage, this function will be called once (or not at
    # all) just before the call to initQuda(), but it's valid to call
    # it any number of times at any point during execution.  Prior to
    # the first time it's called, the parameters take default values
    # as indicated below.
    #
    # @param verbosity  Default verbosity, ranging from QUDA_SILENT to
    #                   QUDA_DEBUG_VERBOSE.  Within a solver, this
    #                   parameter is overridden by the "verbosity"
    #                   member of QudaInvertParam.  The default value
    #                   is QUDA_SUMMARIZE.
    #
    # @param prefix     String to prepend to all messages from QUDA.  This
    #                   defaults to the empty string (""), but you may
    #                   wish to specify something like "QUDA: " to
    #                   distinguish QUDA's output from that of your
    #                   application.
    #
    # @param outfile    File pointer (such as stdout, stderr, or a handle
    #                   returned by fopen()) where messages should be
    #                   printed.  The default is stdout.
    #
    void setVerbosityQuda(QudaVerbosity verbosity, const char prefix[],
                          FILE *outfile)

    #
    # initCommsGridQuda() takes an optional "rank_from_coords" argument that
    # should be a pointer to a user-defined function with this prototype.
    #
    # @param coords  Node coordinates
    # @param fdata   Any auxiliary data needed by the function
    # @return        MPI rank or QMP node ID cooresponding to the node coordinates
    #
    # @see initCommsGridQuda
    #
    ctypedef int (*QudaCommsMap)(const int *coords, void *fdata)

    #
    # @param mycomm User provided MPI communicator in place of MPI_COMM_WORLD
    #

    void qudaSetCommHandle(void *mycomm)

    #
    # Declare the grid mapping ("logical topology" in QMP parlance)
    # used for communications in a multi-GPU grid.  This function
    # should be called prior to initQuda().  The only case in which
    # it's optional is when QMP is used for communication and the
    # logical topology has already been declared by the application.
    #
    # @param nDim   Number of grid dimensions.  "4" is the only supported
    #               value currently.
    #
    # @param dims   Array of grid dimensions.  dims[0]*dims[1]*dims[2]*dims[3]
    #               must equal the total number of MPI ranks or QMP nodes.
    #
    # @param func   Pointer to a user-supplied function that maps coordinates
    #               in the communication grid to MPI ranks (or QMP node IDs).
    #               If the pointer is NULL, the default mapping depends on
    #               whether QMP or MPI is being used for communication.  With
    #               QMP, the existing logical topology is used if it's been
    #               declared.  With MPI or as a fallback with QMP, the default
    #               ordering is lexicographical with the fourth ("t") index
    #               varying fastest.
    #
    # @param fdata  Pointer to any data required by "func" (may be NULL)
    #
    # @see QudaCommsMap
    #

    void initCommsGridQuda(int nDim, const int *dims, QudaCommsMap func, void *fdata)

    #
    # Initialize the library.  This is a low-level interface that is
    # called by initQuda.  Calling initQudaDevice requires that the
    # user also call initQudaMemory before using QUDA.
    #
    # @param device CUDA device number to use.  In a multi-GPU build,
    #               this parameter may either be set explicitly on a
    #               per-process basis or set to -1 to enable a default
    #               allocation of devices to processes.
    #
    void initQudaDevice(int device)

    #
    # Initialize the library persistant memory allocations (both host
    # and device).  This is a low-level interface that is called by
    # initQuda.  Calling initQudaMemory requires that the user has
    # previously called initQudaDevice.
    #
    void initQudaMemory()

    #
    # Initialize the library.  This function is actually a wrapper
    # around calls to initQudaDevice() and initQudaMemory().
    #
    # @param device  CUDA device number to use.  In a multi-GPU build,
    #                this parameter may either be set explicitly on a
    #                per-process basis or set to -1 to enable a default
    #                allocation of devices to processes.
    #
    void initQuda(int device)

    #
    # Finalize the library.
    #
    void endQuda()

    #
    # @brief update the radius for halos.
    # @details This should only be needed for automated testing when
    # different partitioning is applied within a single run.
    #
    void updateR()

    #
    # A new QudaGaugeParam should always be initialized immediately
    # after it's defined (and prior to explicitly setting its members)
    # using this function.  Typical usage is as follows:
    #
    #   QudaGaugeParam gauge_param = newQudaGaugeParam()
    #
    QudaGaugeParam newQudaGaugeParam()

    #
    # A new QudaInvertParam should always be initialized immediately
    # after it's defined (and prior to explicitly setting its members)
    # using this function.  Typical usage is as follows:
    #
    #   QudaInvertParam invert_param = newQudaInvertParam()
    #
    QudaInvertParam newQudaInvertParam()

    #
    # A new QudaMultigridParam should always be initialized immediately
    # after it's defined (and prior to explicitly setting its members)
    # using this function.  Typical usage is as follows:
    #
    #   QudaMultigridParam mg_param = newQudaMultigridParam()
    #
    QudaMultigridParam newQudaMultigridParam()

    #
    # A new QudaEigParam should always be initialized immediately
    # after it's defined (and prior to explicitly setting its members)
    # using this function.  Typical usage is as follows:
    #
    #   QudaEigParam eig_param = newQudaEigParam()
    #
    QudaEigParam newQudaEigParam()

    #
    # A new QudaGaugeObservableParam should always be initialized
    # immediately after it's defined (and prior to explicitly setting
    # its members) using this function.  Typical usage is as follows:
    #
    #   QudaGaugeObservalbeParam obs_param = newQudaGaugeObservableParam();
    #
    QudaGaugeObservableParam newQudaGaugeObservableParam()

    #
    # A new QudaGaugeSmearParam should always be initialized
    # immediately after it's defined (and prior to explicitly setting
    # its members) using this function.  Typical usage is as follows:
    #
    #   QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
    #
    QudaGaugeSmearParam newQudaGaugeSmearParam()

    #
    # A new QudaBLASParam should always be initialized immediately
    # after it's defined (and prior to explicitly setting its members)
    # using this function.  Typical usage is as follows:
    #
    #   QudaBLASParam blas_param = newQudaBLASParam()
    #
    QudaBLASParam newQudaBLASParam()

    #
    # Print the members of QudaGaugeParam.
    # @param param The QudaGaugeParam whose elements we are to print.
    #
    void printQudaGaugeParam(QudaGaugeParam *param)

    #
    # Print the members of QudaInvertParam.
    # @param param The QudaInvertParam whose elements we are to print.
    #
    void printQudaInvertParam(QudaInvertParam *param)

    #
    # Print the members of QudaMultigridParam.
    # @param param The QudaMultigridParam whose elements we are to print.
    #
    void printQudaMultigridParam(QudaMultigridParam *param)

    #
    # Print the members of QudaEigParam.
    # @param param The QudaEigParam whose elements we are to print.
    #
    void printQudaEigParam(QudaEigParam *param)

    #
    # Print the members of QudaGaugeObservableParam.
    # @param param The QudaGaugeObservableParam whose elements we are to print.
    #
    void printQudaGaugeObservableParam(QudaGaugeObservableParam *param)

    #
    # Print the members of QudaBLASParam.
    # @param param The QudaBLASParam whose elements we are to print.
    #
    void printQudaBLASParam(QudaBLASParam *param)

    #
    # Load the gauge field from the host.
    # @param h_gauge Base pointer to host gauge field (regardless of dimensionality)
    # @param param   Contains all metadata regarding host and device storage
    #
    void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)

    #
    # Free QUDA's internal copy of the gauge field.
    #
    void freeGaugeQuda()

    #
    # Free a unique type (Wilson, HISQ fat, HISQ long, smeared) of internal gauge field.
    # @param link_type[in] Type of link type to free up
    #
    void freeUniqueGaugeQuda(QudaLinkType link_type)

    #
    # Free QUDA's internal smeared gauge field.
    #
    void freeGaugeSmearedQuda()

    #
    # Free QUDA's internal two-link gauge field.
    #
    void freeGaugeTwoLinkQuda()

    #
    # Save the gauge field to the host.
    # @param h_gauge Base pointer to host gauge field (regardless of dimensionality)
    # @param param   Contains all metadata regarding host and device storage
    #
    void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param)

    #
    # Load the clover term and/or the clover inverse from the host.
    # Either h_clover or h_clovinv may be set to NULL.
    # @param h_clover    Base pointer to host clover field
    # @param h_cloverinv Base pointer to host clover inverse field
    # @param inv_param   Contains all metadata regarding host and device storage
    #
    void loadCloverQuda(void *h_clover, void *h_clovinv,
                        QudaInvertParam *inv_param)

    #
    # Free QUDA's internal copy of the clover term and/or clover inverse.
    #
    void freeCloverQuda()

    #
    # Perform the solve, according to the parameters set in param.  It
    # is assumed that the gauge field has already been loaded via
    # loadGaugeQuda().
    # @param h_x    Solution spinor field
    # @param h_b    Source spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage and solver parameters
    #
    void lanczosQuda(int k0, int m, void *hp_Apsi, void *hp_r, void *hp_V, void *hp_alpha, void *hp_beta,
                     QudaEigParam *eig_param)

    #
    # Perform the eigensolve. The problem matrix is defined by the invert param, the
    # mode of solution is specified by the eig param. It is assumed that the gauge
    # field has already been loaded via  loadGaugeQuda().
    # @param h_evecs  Array of pointers to application eigenvectors
    # @param h_evals  Host side eigenvalues
    # @param param Contains all metadata regarding the type of solve.
    #
    void eigensolveQuda(void **h_evecs, double_complex *h_evals, QudaEigParam *param)

    #
    # Perform the solve, according to the parameters set in param.  It
    # is assumed that the gauge field has already been loaded via
    # loadGaugeQuda().
    # @param h_x    Solution spinor field
    # @param h_b    Source spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage and solver parameters
    #
    void invertQuda(void *h_x, void *h_b, QudaInvertParam *param)

    #
    # @brief Perform the solve like @invertQuda but for multiple rhs by spliting the comm grid into
    # sub-partitions: each sub-partition invert one or more rhs'.
    # The QudaInvertParam object specifies how the solve should be performed on each sub-partition.
    # Unlike @invertQuda, the interface also takes the host side gauge as input. The gauge pointer and
    # gauge_param are used if for inv_param split_grid[0] * split_grid[1] * split_grid[2] * split_grid[3]
    # is larger than 1, in which case gauge field is not required to be loaded beforehand; otherwise
    # this interface would just work as @invertQuda, which requires gauge field to be loaded beforehand,
    # and the gauge field pointer and gauge_param are not used.
    # @param _hp_x       Array of solution spinor fields
    # @param _hp_b       Array of source spinor fields
    # @param param       Contains all metadata regarding host and device storage and solver parameters
    #
    void invertMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param)

    #
    # Solve for multiple shifts (e.g., masses).
    # @param _hp_x    Array of solution spinor fields
    # @param _hp_b    Source spinor fields
    # @param param  Contains all metadata regarding host and device
    #               storage and solver parameters
    #
    void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param)

    #
    # Setup the multigrid solver, according to the parameters set in param.  It
    # is assumed that the gauge field has already been loaded via
    # loadGaugeQuda().
    # @param param  Contains all metadata regarding host and device
    #               storage and solver parameters
    #
    void* newMultigridQuda(QudaMultigridParam *param)

    #
    # @brief Free resources allocated by the multigrid solver
    # @param mg_instance Pointer to instance of multigrid_solver
    # @param param Contains all metadata regarding host and device
    # storage and solver parameters
    #
    void destroyMultigridQuda(void *mg_instance)

    #
    # @brief Updates the multigrid preconditioner for the new gauge / clover field
    # @param mg_instance Pointer to instance of multigrid_solver
    # @param param Contains all metadata regarding host and device
    # storage and solver parameters, of note contains a flag specifying whether
    # to do a full update or a thin update.
    #
    void updateMultigridQuda(void *mg_instance, QudaMultigridParam *param)

    #
    # @brief Dump the null-space vectors to disk
    # @param[in] mg_instance Pointer to the instance of multigrid_solver
    # @param[in] param Contains all metadata regarding host and device
    # storage and solver parameters (QudaMultigridParam::vec_outfile
    # sets the output filename prefix).
    #
    void dumpMultigridQuda(void *mg_instance, QudaMultigridParam *param)

    #
    # Apply the Dslash operator (D_{eo} or D_{oe}).
    # @param[out] h_out  Result spinor field
    # @param[in] h_in   Input spinor field
    # @param[in] param  Contains all metadata regarding host and device
    #               storage
    # @param[in] parity The destination parity of the field
    #
    void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity)

    #
    # Apply the covariant derivative.
    # @param[out] h_out  Result spinor field
    # @param[in] h_in   Input spinor field
    # @param[in] dir    Direction of application
    # @param[in] param  Metadata for host and device storage
    #
    void covDevQuda(void *h_out, void *h_in, int dir, QudaInvertParam *param)

    #
    # Apply the covariant derivative.
    # @param[out] h_out  Result spinor field
    # @param[in] h_in   Input spinor field
    # @param[in] dir    Direction of application
    # @param[in] sym    Apply forward=2, backward=2 or symmetric=3 shift
    # @param[in] param  Metadata for host and device storage
    #
    void shiftQuda(void *h_out, void *h_in, int dir, int sym, QudaInvertParam *param)

    #
    # Apply the spin-taste operator.
    # @param[out] h_out  Result spinor field
    # @param[in] h_in   Input spinor field
    # @param[in] spin   Spin gamma structure
    # @param[in] taste  Taste gamma structure
    # @param[in] param  Metadata for host and device storage
    #
    void spinTasteQuda(void *h_out, void *h_in, int spin, int taste, QudaInvertParam *param)

    #
    # @brief Perform the solve like @dslashQuda but for multiple rhs by spliting the comm grid into
    # sub-partitions: each sub-partition does one or more rhs'.
    # The QudaInvertParam object specifies how the solve should be performed on each sub-partition.
    # Unlike @invertQuda, the interface also takes the host side gauge as
    # input - gauge field is not required to be loaded beforehand.
    # @param _hp_x       Array of solution spinor fields
    # @param _hp_b       Array of source spinor fields
    # @param param       Contains all metadata regarding host and device storage and solver parameters
    # @param parity      Parity to apply dslash on
    #
    void dslashMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, QudaParity parity)

    #
    # Apply the clover operator or its inverse.
    # @param h_out  Result spinor field
    # @param h_in   Input spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage
    # @param parity The source and destination parity of the field
    # @param inverse Whether to apply the inverse of the clover term
    #
    void cloverQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity, int inverse)

    #
    # Apply the full Dslash matrix, possibly even/odd preconditioned.
    # @param h_out  Result spinor field
    # @param h_in   Input spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage
    #
    void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)

    #
    # Apply M^{\dag}M, possibly even/odd preconditioned.
    # @param h_out  Result spinor field
    # @param h_in   Input spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage
    #
    void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)


    #
    # The following routines are temporary additions used by the HISQ
    # link-fattening code.
    #

    void set_dim(int *)
    void pack_ghost(void **cpuLink, void **cpuGhost, int nFace,
                    QudaPrecision precision)

    void computeKSLinkQuda(void* fatlink, void* longlink, void* ulink, void* inlink,
                           double *path_coeff, QudaGaugeParam *param)

    #
    # Compute two-link field
    #
    # @param[out] twolink computed two-link field
    # @param[in] inlink  the external field
    # @param[in] param  Contains all metadata regarding host and device
    #               storage
    #
    void computeTwoLinkQuda(void *twolink, void *inlink, QudaGaugeParam *param)

    #
    # Either downloads and sets the resident momentum field, or uploads
    # and returns the resident momentum field
    #
    # @param[in,out] mom The external momentum field
    # @param[in] param The parameters of the external field
    #
    void momResidentQuda(void *mom, QudaGaugeParam *param)

    #
    # Compute the gauge force and update the momentum field
    #
    # @param[in,out] mom The momentum field to be updated
    # @param[in] sitelink The gauge field from which we compute the force
    # @param[in] input_path_buf[dim][num_paths][path_length]
    # @param[in] path_length One less that the number of links in a loop (e.g., 3 for a staple)
    # @param[in] loop_coeff Coefficients of the different loops in the Symanzik action
    # @param[in] num_paths How many contributions from path_length different "staples"
    # @param[in] max_length The maximum number of non-zero of links in any path in the action
    # @param[in] dt The integration step size (for MILC this is dt*beta/3)
    # @param[in] param The parameters of the external fields and the computation settings
    #
    int computeGaugeForceQuda(void *mom, void *sitelink, int ***input_path_buf, int *path_length, double *loop_coeff,
                              int num_paths, int max_length, double dt, QudaGaugeParam *qudaGaugeParam)

    #
    # Compute the product of gauge links along a path and add to/overwrite the output field
    #
    # @param[in,out] out The output field to be updated
    # @param[in] sitelink The gauge field from which we compute the products of gauge links
    # @param[in] input_path_buf[dim][num_paths][path_length]
    # @param[in] path_length One less that the number of links in a loop (e.g., 3 for a staple)
    # @param[in] loop_coeff Coefficients of the different loops in the Symanzik action
    # @param[in] num_paths How many contributions from path_length different "staples"
    # @param[in] max_length The maximum number of non-zero of links in any path in the action
    # @param[in] dt The integration step size (for MILC this is dt*beta/3)
    # @param[in] param The parameters of the external fields and the computation settings
    #
    int computeGaugePathQuda(void *out, void *sitelink, int ***input_path_buf, int *path_length, double *loop_coeff,
                             int num_paths, int max_length, double dt, QudaGaugeParam *qudaGaugeParam)

    #
    # Compute the traces of products of gauge links along paths using the resident field
    #
    # @param[in,out] traces The computed traces
    # @param[in] sitelink The gauge field from which we compute the products of gauge links
    # @param[in] path_length The number of links in each loop
    # @param[in] loop_coeff Multiplicative coefficients for each loop
    # @param[in] num_paths Total number of loops
    # @param[in] max_length The maximum number of non-zero of links in any path in the action
    # @param[in] factor An overall normalization factor
    #
    void computeGaugeLoopTraceQuda(double_complex *traces, int **input_path_buf, int *path_length, double *loop_coeff,
                                   int num_paths, int max_length, double factor)

    #
    # Evolve the gauge field by step size dt, using the momentum field
    # I.e., Evalulate U(t+dt) = e(dt pi) U(t)
    #
    # @param gauge The gauge field to be updated
    # @param momentum The momentum field
    # @param dt The integration step size step
    # @param conj_mom Whether to conjugate the momentum matrix
    # @param exact Whether to use an exact exponential or Taylor expand
    # @param param The parameters of the external fields and the computation settings
    #
    void updateGaugeFieldQuda(void* gauge, void* momentum, double dt,
                              int conj_mom, int exact, QudaGaugeParam* param)

    #
    # Apply the staggered phase factors to the gauge field.  If the
    # imaginary chemical potential is non-zero then the phase factor
    # exp(imu/T) will be applied to the links in the temporal
    # direction.
    #
    # @param gauge_h The gauge field
    # @param param The parameters of the gauge field
    #
    void staggeredPhaseQuda(void *gauge_h, QudaGaugeParam *param)

    #
    # Project the input field on the SU(3) group.  If the target
    # tolerance is not met, this routine will give a runtime error.
    #
    # @param gauge_h The gauge field to be updated
    # @param tol The tolerance to which we iterate
    # @param param The parameters of the gauge field
    #
    void projectSU3Quda(void *gauge_h, double tol, QudaGaugeParam *param)

    #
    # Evaluate the momentum contribution to the Hybrid Monte Carlo
    # action.
    #
    # @param momentum The momentum field
    # @param param The parameters of the external fields and the computation settings
    # @return momentum action
    #
    double momActionQuda(void* momentum, QudaGaugeParam* param)

    #
    # Allocate a gauge (matrix) field on the device and optionally download a host gauge field.
    #
    # @param gauge The host gauge field (optional - if set to 0 then the gauge field zeroed)
    # @param geometry The geometry of the matrix field to create (1 - scalar, 4 - vector, 6 - tensor)
    # @param param The parameters of the external field and the field to be created
    # @return Pointer to the gauge field (cast as a void*)
    #
    void* createGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param)

    #
    # Copy the QUDA gauge (matrix) field on the device to the CPU
    #
    # @param outGauge Pointer to the host gauge field
    # @param inGauge Pointer to the device gauge field (QUDA device field)
    # @param param The parameters of the host and device fields
    #
    void  saveGaugeFieldQuda(void* outGauge, void* inGauge, QudaGaugeParam* param)

    #
    # Reinterpret gauge as a pointer to a GaugeField and call destructor.
    #
    # @param gauge Gauge field to be freed
    #
    void destroyGaugeFieldQuda(void* gauge)

    #
    # Compute the clover field and its inverse from the resident gauge field.
    #
    # @param param The parameters of the clover field to create
    #
    void createCloverQuda(QudaInvertParam* param)

    #
    # Compute the clover force contributions from a set of partial
    # fractions stemming from a rational approximation suitable for use
    # within MILC.
    #
    # @param mom Force matrix
    # @param dt Integrating step size
    # @param x Array of solution vectors
    # @param p Array of intermediate vectors
    # @param coeff Array of residues for each contribution (multiplied by stepsize)
    # @param kappa2 -kappa*kappa parameter
    # @param ck -clover_coefficient * kappa / 8
    # @param nvec Number of vectors
    # @param multiplicity Number fermions this bilinear reresents
    # @param gauge Gauge Field
    # @param gauge_param Gauge field meta data
    # @param inv_param Dirac and solver meta data
    #
    void computeCloverForceQuda(void *mom, double dt, void **x, void **p, double *coeff, double kappa2, double ck,
                                int nvector, double multiplicity, void *gauge,
                                QudaGaugeParam *gauge_param, QudaInvertParam *inv_param)

    #
    # Compute the force from a clover or twisted clover determinant or
    # a set of partial fractions stemming from a rational approximation
    # suitable for use from within tmLQCD.
    #
    # @param h_mom Host force matrix
    # @param h_x Array of solution vectors x_i = ( Q^2 + s_i )^{-1} b
    # @param h_x0 Array of source vector necessary to compute the force of a ratio of determinant
    # @param coeff Array of coefficients for the rational approximation or {1.0} for the determinant.
    # @param nvector Number of solution vectors and coefficients
    # @param gauge_param Gauge field meta data
    # @param inv_param Dirac and solver meta data
    # @param detratio if 0 compute the force of a determinant otherwise compute the force from a ratio of determinants
    #
    void computeTMCloverForceQuda(void *h_mom, void **h_x, void **h_x0, double *coeff, int nvector,
                                  QudaGaugeParam *gauge_param, QudaInvertParam *inv_param, int detratio)

    #
    # Compute the naive staggered force.  All fields must be in the same precision.
    #
    # @param mom Momentum field
    # @param dt Integrating step size
    # @param delta Additional scale factor when updating momentum (mom += delta * [force]_TA
    # @param gauge Gauge field (at present only supports resident gauge field)
    # @param x Array of single-parity solution vectors (at present only supports resident solutions)
    # @param gauge_param Gauge field meta data
    # @param invert_param Dirac and solver meta data
    #
    void computeStaggeredForceQuda(void *mom, double dt, double delta, void *gauge, void **x, QudaGaugeParam *gauge_param,
                                   QudaInvertParam *invert_param)

    #
    # Compute the fermion force for the HISQ quark action and integrate the momentum.
    # @param momentum        The momentum field we are integrating
    # @param dt              The stepsize used to integrate the momentum
    # @param level2_coeff    The coefficients for the second level of smearing in the quark action.
    # @param fat7_coeff      The coefficients for the first level of smearing (fat7) in the quark action.
    # @param w_link          Unitarized link variables obtained by applying fat7 smearing and unitarization to the original links.
    # @param v_link          Fat7 link variables.
    # @param u_link          SU(3) think link variables.
    # @param quark           The input fermion field.
    # @param num             The number of quark fields
    # @param num_naik        The number of naik contributions
    # @param coeff           The coefficient multiplying the fermion fields in the outer product
    # @param param.          The field parameters.
    #
    void computeHISQForceQuda(void* momentum,
                              double dt,
                              const double level2_coeff[6],
                              const double fat7_coeff[6],
                              const void* const w_link,
                              const void* const v_link,
                              const void* const u_link,
                              void** quark,
                              int num,
                              int num_naik,
                              double** coeff,
                              QudaGaugeParam* param)

    #
    # @brief Generate Gaussian distributed fields and store in the
    # resident gauge field. We create a Gaussian-distributed su(n)
    # field and exponentiate it, e.g., U = exp(sigma * H), where H is
    # the distributed su(n) field and sigma is the width of the
    # distribution (sigma = 0 results in a free field, and sigma = 1 has
    # maximum disorder).

    # @param seed The seed used for the RNG
    # @param sigma Width of Gaussian distrubution
    #
    void gaussGaugeQuda(unsigned long long seed, double sigma)

    #
    # @brief Generate Gaussian distributed fields and store in the
    # resident momentum field. We create a Gaussian-distributed su(n)
    # field, e.g., sigma * H, where H is the distributed su(n) field
    # and sigma is the width of the distribution (sigma = 0 results
    # in a free field, and sigma = 1 has maximum disorder).
    #
    # @param seed The seed used for the RNG
    # @param sigma Width of Gaussian distrubution
    #
    void gaussMomQuda(unsigned long long seed, double sigma)

    #
    # Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
    # @param[out] Array for storing the averages (total, spatial, temporal)
    #
    void plaqQuda(double plaq[3])

    #
    # @brief Computes the trace of the Polyakov loop of the current resident field
    # in a given direction.

    # @param[out] ploop Trace of the Polyakov loop in direction dir
    # @param[in] dir Direction of Polyakov loop
    #
    void polyakovLoopQuda(double ploop[2], int dir)

    #
    # Performs a deep copy from the internal extendedGaugeResident field.
    # @param Pointer to externally allocated GaugeField
    #
    void copyExtendedResidentGaugeQuda(void *resident_gauge)

    #
    # Performs gaussian/Wuppertal smearing on a given spinor using the gauge field
    # gaugeSmeared, if it exist, or gaugePrecise if no smeared field is present.
    # @param h_in   Input spinor field
    # @param h_out  Output spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage and operator which will be applied to the spinor
    # @param n_steps Number of steps to apply.
    # @param coeff  Width of the Gaussian distribution
    # @param smear_type Gaussian/Wuppertal smearing
    #
    void performFermionSmearQuda(void *h_out, void *h_in, QudaInvertParam *param, const int n_steps, const double coeff,
                                const QudaFermionSmearType smear_type)

    #
    # LEGACY
    # Performs Wuppertal smearing on a given spinor using the gauge field
    # gaugeSmeared, if it exist, or gaugePrecise if no smeared field is present.
    # @param h_out  Result spinor field
    # @param h_in   Input spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage and operator which will be applied to the spinor
    # @param n_steps Number of steps to apply.
    # @param alpha  Alpha coefficient for Wuppertal smearing.
    #
    void performWuppertalnStep(void *h_out, void *h_in, QudaInvertParam *param, unsigned int n_steps, double alpha)

    #
    # LEGACY
    # Performs gaussian smearing on a given spinor using the gauge field
    # gaugeSmeared, if it exist, or gaugePrecise if no smeared field is present.
    # @param h_in   Input spinor field
    # @param h_out  Output spinor field
    # @param param  Contains all metadata regarding host and device
    #               storage and operator which will be applied to the spinor
    # @param n_steps Number of steps to apply.
    # @param omega  Width of the Gaussian distribution
    #
    void performGaussianSmearNStep(void *h_out, void *h_in, QudaInvertParam *param, const int n_steps, const double omega)

    #
    # Performs APE, Stout, or Over Imroved STOUT smearing on gaugePrecise and stores it in gaugeSmeared
    # @param[in] smear_param Parameter struct that defines the computation parameters
    # @param[in,out] obs_param Parameter struct that defines which
    # observables we are making and the resulting observables.
    #
    void performGaugeSmearQuda(QudaGaugeSmearParam *smear_param, QudaGaugeObservableParam *obs_param)

    #
    # Performs Wilson Flow on gaugePrecise and stores it in gaugeSmeared
    # @param[in] smear_param Parameter struct that defines the computation parameters
    # @param[in,out] obs_param Parameter struct that defines which
    # observables we are making and the resulting observables.
    #
    void performWFlowQuda(QudaGaugeSmearParam *smear_param, QudaGaugeObservableParam *obs_param)

    #
    # @brief Calculates a variety of gauge-field observables.  If a
    # smeared gauge field is presently loaded (in gaugeSmeared) the
    # observables are computed on this, else the resident gauge field
    # will be used.
    # @param[in,out] param Parameter struct that defines which
    # observables we are making and the resulting observables.
    #
    void gaugeObservablesQuda(QudaGaugeObservableParam *param)

    #
    # Public function to perform color contractions of the host spinors x and y.
    # @param[in] x pointer to host data
    # @param[in] y pointer to host data
    # @param[out] result pointer to the 16 spin projections per lattice site
    # @param[in] cType Which type of contraction (open, degrand-rossi, etc)
    # @param[in] param meta data for construction of ColorSpinorFields.
    # @param[in] X spacetime data for construction of ColorSpinorFields.
    #
    void contractQuda(const void *x, const void *y, void *result, const QudaContractType cType, QudaInvertParam *param,
                      const int *X)

    #
    # @param[in] x pointer to host data array
    # @param[in] y pointer to host data array
    # @param[out] result pointer to the spin*spin projections per lattice slice site
    # @param[in] cType Which type of contraction (open, degrand-rossi, etc)
    # @param[in] param meta data for construction of ColorSpinorFields.
    # @param[in] src_colors color dilution parameter
    # @param[in] X local lattice dimansions
    # @param[in] source_position source position array
    # @param[in] number of momentum modes
    # @param[in] mom_modes momentum modes
    # @param[in] fft_type Fourier phase factor type (cos, sin or exp{ikx})
    #
    void contractFTQuda(void **x, void **y, void **result, const QudaContractType cType, void *cs_param_ptr,
                        const int src_colors, const int *X, const int *const source_position, const int n_mom,
                        const int *const mom_modes, const QudaFFTSymmType *const fft_type)


    #
    # @brief Gauge fixing with overrelaxation with support for single and multi GPU.
    # @param[in,out] gauge, gauge field to be fixed
    # @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
    # @param[in] Nsteps, maximum number of steps to perform gauge fixing
    # @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
    # @param[in] relax_boost, gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
    # @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when
    # iteration reachs the maximum number of steps defined by Nsteps
    # @param[in] reunit_interval, reunitarize gauge field when iteration count is a multiple of this
    # @param[in] stopWtheta, 0 for MILC criterion and 1 to use the theta value
    # @param[in] param The parameters of the external fields and the computation settings
    #
    int computeGaugeFixingOVRQuda(void *gauge, const unsigned int gauge_dir, const unsigned int Nsteps,
                                  const unsigned int verbose_interval, const double relax_boost, const double tolerance,
                                  const unsigned int reunit_interval, const unsigned int stopWtheta, QudaGaugeParam *param)

    #
    # @brief Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
    # @param[in,out] gauge, gauge field to be fixed
    # @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
    # @param[in] Nsteps, maximum number of steps to perform gauge fixing
    # @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
    # @param[in] alpha, gauge fixing parameter of the method, most common value is 0.08
    # @param[in] autotune, 1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
    # @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when
    # iteration reachs the maximum number of steps defined by Nsteps
    # @param[in] stopWtheta, 0 for MILC criterion and 1 to use the theta value
    # @param[in] param The parameters of the external fields and the computation settings
    #
    int computeGaugeFixingFFTQuda(void *gauge, const unsigned int gauge_dir, const unsigned int Nsteps,
                                  const unsigned int verbose_interval, const double alpha, const unsigned int autotune,
                                  const double tolerance, const unsigned int stopWtheta, QudaGaugeParam *param)

    #
    # @brief Strided Batched GEMM
    # @param[in] arrayA The array containing the A matrix data
    # @param[in] arrayB The array containing the B matrix data
    # @param[in] arrayC The array containing the C matrix data
    # @param[in] native boolean to use either the native or generic version
    # @param[in] param The data defining the problem execution.
    #
    void blasGEMMQuda(void *arrayA, void *arrayB, void *arrayC, QudaBoolean native, QudaBLASParam *param)

    #
    # @brief Strided Batched in-place matrix inversion via LU
    # @param[in] Ainv The array containing the A inverse matrix data
    # @param[in] A The array containing the A matrix data
    # @param[in] use_native Boolean to use either the native or generic version
    # @param[in] param The data defining the problem execution.
    #
    void blasLUInvQuda(void *Ainv, void *A, QudaBoolean use_native, QudaBLASParam *param)

    #
    # @brief Flush the chronological history for the given index
    # @param[in] index Index for which we are flushing
    #
    void flushChronoQuda(int index)


    #
    # Create deflation solver resources.
    #
    #
    void* newDeflationQuda(QudaEigParam *param)

    #
    # Free resources allocated by the deflated solver
    #
    void destroyDeflationQuda(void *df_instance)

    void setMPICommHandleQuda(void *mycomm)

    # Parameter set for quark smearing operations
    ctypedef struct QudaQuarkSmearParam:
        pass

    #
    # Performs two-link Gaussian smearing on a given spinor (for staggered fermions).
    # @param[in,out] h_in Input spinor field to smear
    # @param[in] smear_param   Contains all metadata the operator which will be applied to the spinor
    #
    void performTwoLinkGaussianSmearNStep(void *h_in, QudaQuarkSmearParam *smear_param)

    #
    # @brief Performs contractions between a set of quark fields and
    # eigenvectors of the 3-d Laplace operator.
    # @param[in,out] host_sinks An array representing the inner
    # products between the quark fields and the eigen-vector fields.
    # Ordered as [nQuark][nEv][Lt][nSpin][complexity].
    # @param[in] host_quark Array of quark fields we are taking the inner over
    # @param[in] n_quark Number of quark fields
    # @param[in] tile_quark Tile size for quark fields (batch size)
    # @param[in] host_evec Array of eigenvectors we are taking the inner over
    # @param[in] n_evec Number of eigenvectors
    # @param[in] tile_evec Tile size for eigenvectors (batch size)
    # @param[in] inv_param Meta-data structure
    # @param[in] X Lattice dimensions
    #
    void laphSinkProject(double_complex *host_sinks, void **host_quark, int n_quark, int tile_quark,
                        void **host_evec, int nevec, int tile_evec, QudaInvertParam *inv_param, const int X[4])



