from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy


def dynamic(library, include_path, library_path):
    return Extension(
        name=f"pyquda_plugins.py{library}",
        sources=[f"pyquda_plugins/py{library}/py{library}.pyx"],
        include_dirs=[include_path, numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        library_dirs=[library_path],
        libraries=[library],
        extra_link_args=[f"-Wl,-rpath={library_path}"],
        language="c",
    )


extensions = cythonize(
    [
        dynamic("gwu", "/home/jiangxy/xqcd/gwu-qcd-ck/kentucky", "/home/jiangxy/xqcd/gwu-qcd-ck/build"),
    ],
    language_level="3",
)

setup(ext_modules=extensions)
