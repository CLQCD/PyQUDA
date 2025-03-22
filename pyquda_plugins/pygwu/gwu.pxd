cdef extern from "interface_overlap_inverter.h":
    void gwu_init_machine()
    void gwu_shutdown_machine()
    void gwu_invert_overlap(void *propag_in, void *source_in, void *links_in,
                            int *latt_size, double kappa, int num_mass, double *masses, double tol, int maxiter,
                            double ov_ploy_prec, int ov_use_fp32, int ov_test, int one_minus_half_d,
                            int hw_eignum, void *hw_eigvals, void *hw_eigvecs, double hw_eigprec,
                            int ov_eignum, void *ov_eigvals, void *ov_eigvecs, double ov_eigprec)