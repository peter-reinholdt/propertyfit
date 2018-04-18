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


assert len(ref_files) == len(field_files)
for i in range(len(ref_files)):
    try:
        s_ref = structure()
        s_ref.load_h5(ref_files[i])
        s_field = structure()
        s_field.load_h5(field_files[i])
        ref_structures.append(s_ref)
        field_structures.append(s_field)
    except Exception as e:
        print("Warning, recieved an exception {ex}. Ignoring bad structure, please check the files {r} and {f}".format(ex=e, r=ref_files[i], f=field_files[i]))



#use partial to wrap cost function, so we only need a single atest argument (and not constraints, structures)
#then we can call fun(atest) instead of isopol_cost_function(atest, structures, fieldstructures, constraints)

#read initial parameters from a0
a0 = np.zeros(con.nparametersa)
con.restraint = 1.0


fun = functools.partial(isopol_cost_function, structures=ref_structures, fieldstructures=field_structures, constraints=con)
res = minimize(fun, x0=a0, method='SLSQP')

print(res)
print("\n========================================================\n")
print("Final result:")
for a in con.expand_a(res.x):
    print(a[0,0])
