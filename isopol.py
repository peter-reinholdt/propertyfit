#!/usr/bin/env python

import chargefit
import polfit
import scipy.optimize as opt
import numpy as np
import glob

rs = [chargefit.load_file("test/VAL_0.fchk.s") for i in range(6)]
fs = [chargefit.load_file(f) for f in glob.glob("test/field/*.fchk.s")]

a = np.random.rand(rs[0].natoms)

def fun(alpha):
    #todo: symmetry, restrictions on caps?
    return polfit.cost_alpha_iso(rs, fs, alpha)*1000

res = opt.minimize(fun, x0=a, method="slsqp")
print(res)
