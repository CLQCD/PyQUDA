import os

test_dir = os.path.dirname(os.path.abspath(__file__))

try:
    import pyquda
except ModuleNotFoundError:
    import sys

    pyquda_dir = os.path.abspath(os.path.join(test_dir, ".."))
    sys.path.insert(1, pyquda_dir)
    import pyquda
finally:
    print(f"PYQUDA: You are using {pyquda.__file__} as pyquda")
