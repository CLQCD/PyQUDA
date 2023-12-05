import os
from distutils.core import Extension, setup
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext


class build_ext_no_suffix(build_ext):
    def get_ext_filename(self, ext_name):
        ext_path = ext_name.split(".")
        ext_suffix = ".so"
        return os.path.join(*ext_path) + ext_suffix


extensions = [
    Extension(
        name="pyquda.quda.lib.libquda",
        sources=["pyquda/quda/lib/quda.cpp"],
        language="c++",
    )
]

extensions += cythonize(
    [
        Extension(
            name="pyquda.pointer",
            sources=["pyquda/src/pointer.pyx"],
            language="c",
        ),
        Extension(
            name="pyquda.pyquda",
            sources=["pyquda/src/pyquda.pyx"],
            include_dirs=["pyquda/quda/include"],
            library_dirs=["pyquda/quda/lib"],
            extra_link_args=["-Wl,--no-as-needed", "-lquda"],
            language="c",
        ),
        Extension(
            name="pyquda.malloc_pyquda",
            sources=["pyquda/src/malloc_pyquda.pyx"],
            include_dirs=["pyquda/quda/include"],
            library_dirs=["pyquda/quda/lib"],
            extra_link_args=["-Wl,--no-as-needed", "-lquda"],
            language="c++",
        ),
    ],
    language_level="3",
)

setup(ext_modules=extensions, cmdclass={"build_ext": build_ext_no_suffix})
