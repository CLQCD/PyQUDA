from argparse import ArgumentParser
from distutils import log
from distutils.core import Distribution
import logging
import os
import shutil
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import sys
import tempfile

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

from Cython.Build import cythonize
import numpy


def dynamic(lib, header, include_path, library_path):
    from pyquda_plugins.plugin_pyx import build_plugin_pyx

    build_plugin_pyx(os.path.dirname(__file__), lib, header, include_path)
    return cythonize(
        [
            Extension(
                name=f"pyquda_plugins.py{lib}._py{lib}",
                sources=[os.path.join(f"pyquda_plugins/py{lib}/src/_py{lib}.pyx")],
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


def build_and_install(module):
    dist = Distribution(
        {
            "ext_modules": [module],
            "packages": ["pyquda_plugins"],
            "package_data": {"pyquda_plugins": ["*/*.pyi", "*/src/*.pyx", "*/src/*.pxd"]},
            "script_name": "__main__",
        }
    )

    build_temp = tempfile.mkdtemp(prefix="pyquda_plugins_")
    try:
        # log.info("running build_py")
        # _build_py = build_py(dist)
        # _build_py.build_lib = build_temp
        # _build_py.ensure_finalized()
        # _build_py.run()
        log.info("running build_ext")
        _build_ext = build_ext(dist)
        _build_ext.build_lib = build_temp
        _build_ext.build_temp = build_temp
        _build_ext.inplace = True
        _build_ext.ensure_finalized()
        _build_ext.run()
        # if not inplace:
        #     log.info("running install_lib")
        #     _install_lib = install_lib(dist)
        #     _install_lib.inplace = inplace
        #     _install_lib.ensure_finalized()
        #     _install_lib.run()
    finally:
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
    args = parser.parse_args()
    cwd = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    build_and_install(dynamic(args.lib, args.header, args.include_path, args.library_path))
    os.chdir(cwd)


if __name__ == "__main__":
    main()
