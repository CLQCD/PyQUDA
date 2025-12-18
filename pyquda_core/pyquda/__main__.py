from argparse import ArgumentParser

from . import init


def main():
    parser = ArgumentParser(prog="pyquda", description="PyQUDA initializer", epilog="")
    parser.add_argument("script")
    parser.add_argument(
        "-g",
        "--grid",
        "--geom",
        nargs=4,
        type=int,
        help="GPU grid size used to split the lattice",
        metavar=("Gx", "Gy", "Gz", "Gt"),
    )
    parser.add_argument(
        "-l",
        "--latt",
        nargs=4,
        type=int,
        help="Lattice size used to set the default GPU grid",
        metavar=("Lx", "Ly", "Lz", "Lt"),
    )
    parser.add_argument(
        "-m",
        "--grid-map",
        default="default",
        choices=("default", "reversed", "shared"),
        help="Grid mapping used for PyQUDA (default: default)",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default="cupy",
        choices=("numpy", "cupy", "torch", "dpnp"),
        help="Array backend used for PyQUDA (default: cupy)",
    )
    parser.add_argument(
        "-t",
        "--backend-target",
        default="cuda",
        choices=("cpu", "cuda", "hip", "sycl"),
        help="Array backend target used for PyQUDA (default: cuda)",
    )
    parser.add_argument(
        "--no-init-quda",
        action="store_true",
        help="Don't initialize the QUDA library",
    )
    parser.add_argument(
        "-p",
        "--resource-path",
        help="(default: QUDA_RESOURCE_PATH)",
        metavar="QUDA_RESOURCE_PATH",
    )
    args = parser.parse_args()
    init(
        args.grid,
        args.latt,
        args.grid_map,
        args.backend,
        args.backend_target,
        not args.no_init_quda,
        resource_path=args.resource_path,
    )
    exec(open(args.script).read(), globals(), globals())


if __name__ == "__main__":
    main()
