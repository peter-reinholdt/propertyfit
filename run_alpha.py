#!/usr/bin/env python


import sys
import numpy as np
import functools
from scipy.optimize import minimize
from propertyfit.structures import structure, constraints
from propertyfit.costfunctions import isopol_cost_function
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--h5-file-list', dest='h5_filelist', type=str, help='Read which h5 files to use from a file', required=True)
parser.add_argument('--topology', dest='top', type=str, help='Provide file for information about symmetry-equivalent atoms and more.', required=True)
parser.add_argument('--restraint', dest='restraint', type=float, default=0.0, help='Strength of harmonic restraint towards charges from topology file')
parser.add_argument('--method', dest='method', default='slsqp', help='Which optimizer to use')
parser.add_argument('--weights', dest='weights', type=str, help='Weights to use in optimization')
parser.add_argument('-o', dest='output', type=str, default='alphas.dat', help='Name of file to write polarizabilities to')


args = parser.parse_args()
con = constraints(args.top)
files = np.loadtxt(args.h5_filelist, dtype=str)
ref_files = files[:,0]
field_files = files[:,1]
ref_structures = []
field_structures = []

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
a0 = con.expand_a(con.a0)
con.restraint = args.restraint

fun = functools.partial(isopol_cost_function, structures=ref_structures, fieldstructures=field_structures, constraints=con)
constraints = [{'type': 'ineq', 'fun': lambda x: np.min(x)}]
res = minimize(fun, x0=a0, method=args.method, constraints=constraints, tol=1e-12, options={'maxiter':1000})

print(res)
print("\n========================================================\n")
print("Final result:")
with open(args.output, "w") as f:
    for a in con.expand_a(res.x):
        print(a[0,0])
        f.write("{}\n".format(a[0,0]))
