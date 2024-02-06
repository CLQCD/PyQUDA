from argparse import ArgumentParser

from . import init


def main():
    parser = ArgumentParser(prog="pyquda", description="PyQuda initializer", epilog="Text at the bottom of help")
    parser.add_argument("script")
    parser.add_argument(
        "-g",
        "--grid",
        nargs=4,
        type=int,
        required=True,
        help="Grid of how GPUs are arranged",
        metavar=("Gx", "Gy", "Gz", "Gt"),
    )
    parser.add_argument(
        "-b", "--backend", default="cupy", choices=("cupy", "torch"), help="CUDA backend of PyQuda (default: cupy)"
    )
    parser.add_argument(
        "-p",
        "--resource-path",
        help="(default: QUDA_RESOURCE_PATH)",
        metavar="QUDA_RESOURCE_PATH",
    )
    args = parser.parse_args()
    init(args.grid, args.backend, args.resource_path)
    exec(open(args.script).read())


if __name__ == "__main__":
    main()
