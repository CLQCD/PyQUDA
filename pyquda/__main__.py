from argparse import ArgumentParser

from . import init


def main():
    parser = ArgumentParser(prog="pyquda", description="PyQuda initializer", epilog="Text at the bottom of help")
    parser.add_argument("script")
    parser.add_argument(
        "-g",
        "--grid",
        nargs=4,
        default=[1, 1, 1, 1],
        type=int,
        help="Grid of how GPUs are arranged",
        metavar=("Gx", "Gy", "Gz", "Gt"),
    )
    parser.add_argument(
        "-l",
        "--latt",
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
        help="Lattice t boundary used as the default (required if -l/--latt is set)",
    )
    parser.add_argument(
        "-a",
        "--anisotropy",
        type=float,
        help="Lattice anisotropy used as the default (required if -l/--latt is set)",
        metavar="xi",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default="cupy",
        choices=("cupy", "torch"),
        help="CUDA backend of PyQuda (default: cupy)",
    )
    parser.add_argument(
        "-p",
        "--resource-path",
        help="(default: QUDA_RESOURCE_PATH)",
        metavar="QUDA_RESOURCE_PATH",
    )
    parser.add_argument(
        "--enable-mps",
        help="(default: QUDA_ENABLE_MPS)",
        metavar="QUDA_ENABLE_MPS",
    )
    args = parser.parse_args()
    init(
        args.grid,
        args.latt,
        args.t_boundary,
        args.anisotropy,
        backend=args.backend,
        resource_path=args.resource_path,
        enable_mps=args.enable_mps,
    )
    exec(open(args.script).read())


if __name__ == "__main__":
    main()
