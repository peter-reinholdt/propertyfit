#!/usr/bin/env python

import chargefit
import polfit
import scipy.optimize as opt
import numpy as np
import glob
import sys
import re

aafolder=sys.argv[1]

#find and match "unfielded" to "fielded" versions of a particular structure
frsx = glob.glob(aafolder+"/*.fchk.s")
frs = []
ffs = []
for i in frsx:
    x = re.split("/|\.", i)[-3]
    print(x)
    for j in glob.glob(aafolder+"/field/*"+x+"_*.fchk.s"):
        ffs.append(j)
        frs.append(i)

rs = [chargefit.load_file(f) for f in frs]
fs = [chargefit.load_file(f) for f in ffs]

a = np.random.rand(rs[0].natoms)

def fun(alpha):
    #todo: symmetry, restrictions on caps?
    res =  polfit.cost_alpha_iso(rs, fs, alpha)*2625.5002
    print(res)
    return res

res = opt.minimize(fun, x0=a, method="bfgs")
print(res)
