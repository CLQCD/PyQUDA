from argparse import ArgumentParser
from distutils import log
from distutils.core import Distribution
import logging
import os
import shutil
from setuptools import Extension

from distutils.command.build_py import build_py
from distutils.command.build_ext import build_ext
from distutils.command.install_lib import install_lib
import sys
import tempfile

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

from Cython.Build import cythonize
import numpy


def dynamic(lib, header, include_path, library_path):
    from pyquda_plugins.plugin_pyx import Plugin

    plugins_root = os.path.dirname(__file__)
    plugin = Plugin(lib, header, include_path)
    plugin.write(plugins_root)
    source = os.path.join(plugins_root, f"py{lib}", "src", f"_py{lib}.pyx")
    return cythonize(
        [
            Extension(
                name=f"pyquda_plugins.py{lib}._py{lib}",
                sources=[source],
                include_dirs=[include_path, numpy.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                library_dirs=[library_path],
                libraries=[lib],
                extra_link_args=[f"-Wl,-rpath={library_path}"],
                language="c",
            )
        ],
        language_level=3,
    )[0]


def build_and_install(lib, module, inplace):
    build_temp = tempfile.mkdtemp(prefix="pyquda_plugins_")
    dist = Distribution(
        {
            "ext_modules": [module],
            "packages": ["pyquda_plugins"],
            "package_data": {"pyquda_plugins": ["*/*.pyi", "*/src/*.pyx", "*/src/*.pxd"]},
            "script_name": "__main__",
        }
    )

    log.info("running build_py")
    _build_py = build_py(dist)
    _build_py.build_lib = build_temp
    _build_py.ensure_finalized()
    _build_py.run()
    log.info("running build_ext")
    _build_ext = build_ext(dist)
    _build_ext.build_temp = build_temp
    _build_ext.inplace = inplace
    _build_ext.ensure_finalized()
    _build_ext.run()
    if not inplace:
        log.info("running install_lib")
        _install_lib = install_lib(dist)
        _install_lib.inplace = inplace
        _install_lib.ensure_finalized()
        _install_lib.run()

    # if not inplace:
    #     site_packages = sysconfig.get_path("purelib")
    #     plugins_root = os.path.join(site_packages, "pyquda_plugins")
    #     ext = cmd.get_outputs()
    #     for src, module in zip(ext, module_list):
    #         dst = "/tmp"
    #         log.info(f"copying {src} -> {dst}")
    #         # shutil.copy2(src, dst)

    shutil.rmtree(build_temp)


def main():
    parser = ArgumentParser(prog="pyquda_plugins", description="PyQUDA plugins installer", epilog="")
    parser.add_argument(
        "-l",
        "--lib",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--header",
        required=True,
    )
    parser.add_argument(
        "-L",
        "--library_path",
        required=True,
    )
    parser.add_argument(
        "-I",
        "--include_path",
        required=True,
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
    )
    args = parser.parse_args()
    build_and_install(args.lib, dynamic(args.lib, args.header, args.include_path, args.library_path), args.inplace)


if __name__ == "__main__":
    main()
