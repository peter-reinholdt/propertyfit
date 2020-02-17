#!/usr/bin/env python

import sys
from propertyfit.utilities import load_qmfiles

fchk = sys.argv[1]
s = load_qmfiles(fchk)[0]

s.compute_grid()
s.compute_rinvmat()
s.compute_xyzmat()
s.compute_qm_esp()
s.save_h5(fchk + ".h5")
