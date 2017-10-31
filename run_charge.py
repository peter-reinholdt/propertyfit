#!/usr/bin/env python


import sys
import numpy as np
import functools
from scipy.optimize import minimize
from propertyfit.structures import structure, constraints
from propertyfit.costfunctions import charge_cost_function


if len(sys.argv) < 3:
    print("Usage ./run_charge.py constraintsfile locationfile")
    exit()


constraintsfile = sys.argv[1]
locationfile    = sys.argv[2]
files = np.loadtxt(locationfile, dtype=str)

#create constraints object
con = constraints(constraintsfile)

#load structure objects as defined from locationfile
structures = []
for fname in files:
    s = structure()
    s.load_h5(fname)
    structures.append(s)

#use partial to wrap cost function, so we only need a single qtest argument (and not constraints, structures)
#then we can call fun(qtest) instead of charge_cost_function(qtest, structures, constraints)

q0 = con.q0

fun = functools.partial(charge_cost_function, structures=structures, constraints=con)
res = minimize(fun, x0=q0, method='SLSQP')

print(res)
print("\n========================================================\n")
print("Final result:")
for q in con.expand_q(res.x):
    print(q)
