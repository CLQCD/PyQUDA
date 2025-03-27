from Cython.Build import cythonize
from importlib import import_module
import os
from setuptools import Extension, setup

import numpy


def dynamic(lib, header, include_path, library_path):
    import_module(f"_{lib}").setup(lib, header).write(os.path.dirname(__file__))
    return Extension(
        name=f"pyquda_plugins.py{lib}",
        sources=[f"pyquda_plugins/py{lib}.pyx"],
        include_dirs=[include_path, numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        library_dirs=[library_path],
        libraries=[lib],
        extra_link_args=[f"-Wl,-rpath={library_path}"],
        language="c",
    )


if "PYQUDA_PLUGINS" in os.environ:
    libs = os.environ["PYQUDA_PLUGINS"].split(",")
    for i, lib in enumerate(libs):
        lib = lib.upper()
        libs[i] = dynamic(
            lib.lower(),
            os.environ[f"{lib}_HEADER"],
            os.environ[f"{lib}_INCLUDE_PATH"],
            os.environ[f"{lib}_LIBRARY_PATH"],
        )
    setup(ext_modules=cythonize(libs, language_level="3"))
