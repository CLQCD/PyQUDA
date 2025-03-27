from pyquda_plugins.plugin_pyx import Plugin


def bind(lib: str, header: str):
    gwu = Plugin(lib, header)

    gwu.function("gwu_init_machine", "init", [("int *", "latt_size")])
    gwu.function("gwu_shutdown_machine", "shutdown", [])
    gwu.function("gwu_build_hw", "build_hw", [("void *", "links_in"), ("double", "kappa")])
    gwu.function(
        "gwu_load_hw_eigen",
        "load_hw_eigen",
        [("int", "hw_eignum"), ("double", "hw_eigprec"), ("double complex *", "hw_eigvals"), ("void *", "hw_eigvecs")],
    )
    gwu.function("gwu_build_ov", "build_ov", [("double", "ov_poly_prec"), ("int", "ov_use_fp32")])
    gwu.function(
        "gwu_load_ov_eigen",
        "load_ov_eigen",
        [("int", "ov_eignum"), ("double", "ov_eigprec"), ("double complex *", "ov_eigvals"), ("void *", "ov_eigvecs")],
    )
    gwu.function(
        "gwu_build_hw_eigen",
        "build_hw_eigen",
        [
            ("int", "hw_eignum"),
            ("double", "hw_eigprec"),
            ("int", "hw_extra_krylov"),
            ("int", "maxiter"),
            ("int", "chebyshev_order"),
            ("double", "chebyshev_cut"),
            ("int", "iseed"),
        ],
    )
    gwu.function(
        "gwu_invert_overlap",
        "invert_overlap",
        [
            ("void *", "propag_in"),
            ("void *", "source_in"),
            ("int", "num_mass"),
            ("double *", "masses"),
            ("double", "tol"),
            ("int", "maxiter"),
            ("int", "one_minus_half_d"),
            ("int", "mode"),
        ],
    )

    return gwu
