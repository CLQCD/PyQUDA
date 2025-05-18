import os
from check_pyquda import chroma

os.chdir(os.path.dirname(__file__))
os.makedirs("data", exist_ok=True)
assert chroma("test_wilson.ini.xml") == 0  # pt_prop_0
assert chroma("test_clover.ini.xml") == 0  # pt_prop_1
chroma("test_hisq.ini.xml")  # pt_prop_2
assert chroma("test_clover_isotropic.ini.xml") == 0  # pt_prop_3
assert chroma("test_gaussian.ini.xml") == 0  # pt_prop_4
assert chroma("test_smear.ini.xml") == 0  # ape.lime stout.lime hyp.lime
assert chroma("test_wflow.ini.xml") == 0  # wflow.lime
assert chroma("test_gfix.ini.xml") == 0  # wflow.lime
