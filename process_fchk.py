#!/usr/bin/env python
import chargefit
import sys

fchk = sys.argv[1]
try:
    structure = chargefit.load_qmfiles(fchk)[0]
    structure.compute_grid()
    structure.compute_rinvmat()
    structure.compute_xyzmat()
    structure.compute_qm_esp()
    structure.save(fchk+".s")
except Exception as e:
    print("Ooops!")
    print(e)
