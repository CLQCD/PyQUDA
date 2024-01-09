import os
from distutils.core import Extension, setup
from Cython.Build import cythonize

_STATIC = False
if "QUDA_PATH" in os.environ:
    libquda_path = os.path.join(os.path.realpath(os.environ["QUDA_PATH"]), "lib")
    if os.path.exists(os.path.join(libquda_path, "libquda.so")):
        pass
    elif os.path.exists(os.path.join(libquda_path, "libquda.a")):
        _STATIC = True
    else:
        raise RuntimeError("Cannot find libquda.so or libquda.a in QUDA_PATH/lib")
else:
    ld_library_path = [os.path.realpath(_path) for _path in os.environ["LD_LIBRARY_PATH"].strip().split(":")]
    for libquda_path in ld_library_path:
        if os.path.exists(os.path.join(libquda_path, "libquda.so")):
            break
    else:
        raise RuntimeError("Cannot find libquda.so in LD_LIBRARY_PATH")

extensions = cythonize(
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
            library_dirs=[libquda_path],
            libraries=["quda"],
            extra_link_args=[f"-Wl,-rpath={libquda_path}" if not _STATIC else ""],
            language="c",
        ),
        Extension(
            name="pyquda.malloc_pyquda",
            sources=["pyquda/src/malloc_pyquda.pyx"],
            include_dirs=["pyquda/quda/include"],
            library_dirs=[libquda_path],
            libraries=["quda"],
            extra_link_args=[f"-Wl,-rpath={libquda_path}" if not _STATIC else ""],
            language="c++",
        ),
    ],
    language_level="3",
)

setup(ext_modules=extensions)
