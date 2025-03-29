from argparse import ArgumentParser

from . import init


def main():
    parser = ArgumentParser(prog="pyquda", description="PyQUDA initializer", epilog="")
    parser.add_argument("script")
    parser.add_argument(
        "-g",
        "--grid",
        nargs=4,
        type=int,
        help="GPU grid size used to split the lattice",
        metavar=("Gx", "Gy", "Gz", "Gt"),
    )
    parser.add_argument(
        "-l",
        "--latt",
        "--lattice",
        nargs=4,
        type=int,
        help="Lattice size used as the default",
        metavar=("Lx", "Ly", "Lz", "Lt"),
    )
    parser.add_argument(
        "-t",
        "--t-boundary",
        type=int,
        choices=(1, -1),
        help="Lattice t boundary used as the default (required if the lattice size is set)",
    )
    parser.add_argument(
        "-a",
        "--anisotropy",
        type=float,
        help="Lattice anisotropy used as the default (required if the lattice size is set)",
        metavar="xi",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default="cupy",
        choices=("numpy", "cupy", "torch"),
        help="CUDA backend used for PyQUDA (default: cupy)",
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
        args.t_boundary,
        args.anisotropy,
        args.backend,
        not args.no_init_quda,
        resource_path=args.resource_path,
    )
    exec(open(args.script).read(), globals(), globals())


if __name__ == "__main__":
    main()
