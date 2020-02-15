#!/usr/bin/env python


import sys
import numpy as np
import functools
from scipy.optimize import minimize
from propertyfit.structures import structure, constraints
from propertyfit.costfunctions import polarizability_cost_function
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--h5-file-list', dest='h5_filelist', type=str, help='Read which h5 files to use from a file', required=True)
parser.add_argument('--topology', dest='top', type=str, help='Provide file for information about symmetry-equivalent atoms and more.', required=True)
parser.add_argument('--restraint', dest='restraint', type=float, default=0.0, help='Strength of harmonic restraint towards charges from topology file')
parser.add_argument('--method', dest='method', default='slsqp', help='Which optimizer to use')
parser.add_argument('--weights', dest='weights', type=str, help='Weights to use in optimization')
parser.add_argument('--isotropic', dest='isotropic', type=bool, action='store_true', help='Use only isotropic polarizabilities')


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


if args.weights:
    weights = np.loadtxt(args.weights)
else:
    weights = None

con.restraint = args.restraint
parameters = con.get_polarizability_parameter_vector(isotropic=args.isotropic)

for s in structures:
    s.get_rotation_matrices(con)
for s in fieldstructures:
    s.get_rotation_matrices(con)

fun = functools.partial(isopol_cost_function, structures=ref_structures, fieldstructures=field_structures, constraints=con)
res = minimize(fun, x0=parameters, method=args.method, tol=1e-12, options={'maxiter':1000})
polarizabilities_local = con.expand_polarizabilities(res.x)

print(res)

print("=" * 85)
print()
print("Final result:")
print("Polarizabilities (in local axes):")
print("{:>6}   {:<13} {:<13} {:<13} {:<13} {:<13} {:<13}".format("Index", "xx", "xy", "xz", "yy", "yz", "zz"))
for i, polarizability in enumerate(polarizabilities_local):
    print(
        f'{i:>6}: {polarizability[0,0]: 12.10f} {polarizability[0,1]: 12.10f} {polarizability[0,2]: 12.10f} {polarizability[1,1]: 12.10f} {polarizability[1,2]: 12.10f} {polarizability[2,2]: 12.10f}'
    )
