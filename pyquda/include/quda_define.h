/**
   @file quda_define.h
   @brief Macros defined set by the cmake build system.  This file
   should not be edited manually.
 */

/**
 * @def   __COMPUTE_CAPABILITY__
 * @brief This macro sets the target GPU architecture.  Unlike
 * __CUDA_ARCH__, this is defined on host and device.
 */
#define __COMPUTE_CAPABILITY__ 610

/**
 * @def   MAX_MULTI_BLAS_N
 * @brief This macro sets the limit of blas fusion in the multi-blas
 * and multi-reduce kernels
 */
#define MAX_MULTI_BLAS_N 4

/* #undef QUDA_HETEROGENEOUS_ATOMIC */
#ifdef QUDA_HETEROGENEOUS_ATOMIC
/**
 * @def   HETEROGENEOUS_ATOMIC
 * @brief This macro sets whether we are compiling QUDA with heterogeneous atomic
 * support enabled or not
 */
#define HETEROGENEOUS_ATOMIC
#undef QUDA_HETEROGENEOUS_ATOMIC
#endif

/* #undef QUDA_DYNAMIC_CLOVER */
#ifdef QUDA_DYNAMIC_CLOVER
/**
 * @def   DYNAMIC_CLOVER
 * @brief This macro sets whether we are compiling QUDA with dynamic
 * clover inversion support enabled or not
 */
#define DYNAMIC_CLOVER
#undef QUDA_DYNAMIC_CLOVER
#endif

#define QUDA_FLOAT8
#ifdef QUDA_FLOAT8
/**
 * @def FLOAT8
 * @brief This macro set whether float8-ordered fields are enabled or
 * not
 */
#define FLOAT8
#undef QUDA_FLOAT8
#endif
