import os
from distutils.core import Extension, setup
from Cython.Build import cythonize
from pyquda_pyx import build_pyquda_pyx

assert "QUDA_PATH" in os.environ, "QUDA_PATH environment is needed to link against libquda"
quda_path = os.path.realpath(os.environ["QUDA_PATH"])
build_pyquda_pyx(os.path.dirname(__file__), quda_path)
if os.path.exists(os.path.join(quda_path, "lib", "libquda.so")):
    _STATIC = False
elif os.path.exists(os.path.join(quda_path, "lib", "libquda.a")):
    _STATIC = True
else:
    raise RuntimeError(f"Cannot find libquda.so or libquda.a in {os.path.join(quda_path, 'lib')}")

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
            include_dirs=[os.path.join(quda_path, "include")],
            library_dirs=[os.path.join(quda_path, "lib")],
            libraries=["quda"],
            extra_link_args=[f"-Wl,-rpath={os.path.join(quda_path, 'lib')}"] if not _STATIC else None,
            language="c",
        ),
        Extension(
            name="pyquda.malloc_pyquda",
            sources=["pyquda/src/malloc_pyquda.pyx"],
            include_dirs=[os.path.join(quda_path, "include")],
            library_dirs=[os.path.join(quda_path, "lib")],
            libraries=["quda"],
            extra_link_args=[f"-Wl,-rpath={os.path.join(quda_path, 'lib')}"] if not _STATIC else None,
            language="c++",
        ),
    ],
    language_level="3",
)

setup(ext_modules=extensions)
