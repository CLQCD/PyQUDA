cdef extern from "interface_overlap_inverter.h":
    void gwu_init_machine(int *latt_size)
    void gwu_shutdown_machine()
    void gwu_build_hw(void *links_in, double kappa)
    void gwu_load_hw_eigen(int hw_eignum, double hw_eigprec, void *hw_eigvals, void *hw_eigvecs)
    void gwu_build_ov(double ov_poly_prec, int ov_use_fp32)
    void gwu_load_ov_eigen(int ov_eignum, double ov_eigprec, void *ov_eigvals, void *ov_eigvecs)
    void gwu_build_hw_eigen(int hw_eignum, double hw_eigprec, int iseed, int maxiter, int hw_extra_krylov,
                            int chebyshev_order, double chebyshev_cut)
    void gwu_invert_overlap(void *propag_in, void *source_in, int num_mass, double *masses, double tol, int maxiter,
                            int one_minus_half_d, int mode)