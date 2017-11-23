#!/usr/bin/env python


import sys
import numpy as np
import functools
from scipy.optimize import minimize
from propertyfit.structures import structure, constraints
from propertyfit.costfunctions import isopol_cost_function


if len(sys.argv) < 3:
    print("Usage ./run_alpha.py constraintsfile locationfile2")
    print("locationfile2 contains newline separated (refstructure, filestructure)")
    exit()


constraintsfile = sys.argv[1]
locationfile    = sys.argv[2]
files = np.loadtxt(locationfile, dtype=str)
ref_files = files[:,0]
field_files = files[:,1]

#create constraints object
con = constraints(constraintsfile)

#load structure objects as defined from locationfile
ref_structures = []
field_structures = []


for fname in ref_files:
    s = structure()
    s.load_h5(fname)
    ref_structures.append(s)


for fname in field_files:
    s = structure()
    s.load_h5(fname)
    field_structures.append(s)


#use partial to wrap cost function, so we only need a single atest argument (and not constraints, structures)
#then we can call fun(atest) instead of isopol_cost_function(atest, structures, fieldstructures, constraints)

#read initial parameters from q0
a0 = np.zeros(con.nparametersa)



fun = functools.partial(isopol_cost_function, structures=ref_structures, fieldstructures=field_structures, constraints=con)
res = minimize(fun, x0=a0, method='SLSQP')

print(res)
print("\n========================================================\n")
print("Final result:")
for a in con.expand_a(res.x):
    print(a[0,0])
