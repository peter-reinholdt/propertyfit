#!/usr/bin/env python
import chargefit
import sys

fchk = sys.argv[1]
structure = chargefit.load_fchk(fchk)
structure.compute_grid()
structure.compute_rinvmat()
structure.compute_xyzmat()
structure.compute_qm_esp()
structure.save(fchk+".s")
