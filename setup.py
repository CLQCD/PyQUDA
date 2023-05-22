from os import path, environ
from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

VERSION = "0.2.0"
LICENSE = "MIT"
DESCRIPTION = "Python wrapper for quda written in Cython."

ld_library_path = [path.abspath(_path) for _path in environ["LD_LIBRARY_PATH"].strip().split(":")]
for libquda_path in ld_library_path:
    if path.exists(path.join(libquda_path, "libquda.so")):
        break
else:
    raise RuntimeError("Cannot find libquda.so in LD_LIBRARY_PATH environment")

ext_modules = cythonize(
    [
        Extension(
            "pyquda.pyquda",
            ["pyquda/src/pyquda.pyx"],
            language="c",
            include_dirs=["pyquda/include", numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            library_dirs=[libquda_path],
            libraries=["quda"],
        )
    ],
    language_level="3",
)

packages = [
    "pyquda",
    "pyquda.dslash",
    "pyquda.utils",
]
package_dir = {
    "pyquda": "pyquda",
}
package_data = {
    "pyquda": ["*.pyi", "src/*.pxd", "include/*.h"],
}

setup(
    name="PyQuda",
    version=VERSION,
    description=DESCRIPTION,
    author="SaltyChiang",
    author_email="SaltyChiang@users.noreply.github.com",
    packages=packages,
    ext_modules=ext_modules,
    license=LICENSE,
    package_dir=package_dir,
    package_data=package_data,
)
