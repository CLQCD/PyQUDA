import os

test_dir = os.path.dirname(os.path.abspath(__file__))
weak_field = os.path.join(test_dir, "weak_field.lime")

try:
    import pyquda
except ModuleNotFoundError:
    import sys

    pyquda_dir = os.path.abspath(os.path.join(test_dir, ".."))
    sys.path.insert(1, pyquda_dir)
    import pyquda
finally:
    pyquda.getLogger().debug(f"Using {pyquda.__file__} as pyquda")


def chroma(ini_xml: str):
    chroma_path = os.path.abspath(os.path.join(test_dir, "bin", "chroma"))
    ini_xml_path = os.path.abspath(os.path.join(test_dir, ini_xml))
    return os.system(f"{chroma_path} -i {ini_xml_path}")
