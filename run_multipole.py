#!/usr/bin/env python

import sys
import numpy as np
import functools
from scipy.optimize import minimize
from propertyfit.structures import structure, constraints
from propertyfit.costfunctions import multipole_cost_function
import argparse
import cProfile

parser = argparse.ArgumentParser()
parser.add_argument('--h5-files',
                    "-h5",
                    dest="h5",
                    type=str,
                    nargs="+",
                    help="Specify a number of .h5 files with pre-computed potentials.")
parser.add_argument('--h5-file-list', dest='h5_filelist', type=str, help='Read which h5 files to use from a file')
parser.add_argument('--topology',
                    dest='top',
                    type=str,
                    help='Provide file for information about symmetry-equivalent atoms and more.',
                    required=True)
#parser.add_argument('--restraint', dest='restraint', type=float, default=0.0, help='Strength of harmonic restraint towards charges from topology file')
parser.add_argument('--method', dest='method', default='slsqp', help='Which optimizer to use')
parser.add_argument('--weights', dest='weights', type=str, help='Weights to use in optimization')
parser.add_argument('--hydrogen-max-angular-momentum', type=int, default=1)
parser.add_argument('--restraint',
                    dest='restraint',
                    type=float,
                    default=0.0,
                    help='Strength of harmonic restraint towards charges from topology file')

args = parser.parse_args()
#create constraints object
con = constraints(args.top)

#load structure objects as defined from locationfile
structures = []
if args.h5:
    for fname in args.h5:
        s = structure()
        s.load_h5(fname)
        structures.append(s)
if args.h5_filelist:
    with open(args.h5_filelist, "r") as f:
        files = f.read().split()
    for fname in files:
        s = structure()
        s.load_h5(fname)
        structures.append(s)
if len(structures) == 0:
    raise ValueError('Please provide electric potentials via either the --h5-files or --h5-file-list options.')

if args.weights:
    weights = np.loadtxt(args.weights)
else:
    weights = None

con.restraint = args.restraint
parameters = con.get_multipole_parameter_vector(optimize_charges=True,
                                                optimize_dipoles=True,
                                                optimize_quadrupoles=True,
                                                hydrogen_max_angular_momentum=args.hydrogen_max_angular_momentum)
for s in structures:
    s.get_rotation_matrices(con)
fun = functools.partial(multipole_cost_function, structures=structures, constraints=con, weights=weights, calc_jac=True)
res = minimize(fun, x0=parameters, method=args.method, tol=1e-9, jac=True, options={'maxiter': 1000})
charges, dipoles_local, quadrupoles_local = con.expand_parameter_vector(res.x)
print(res)

print("=" * 85)
print()
print("Final result:")
print("Charges:")
print("{:>6}   {:<12}".format("Index", "Charge"))
for i, charge in enumerate(charges):
    print(f'{i:>6}: {charge: 12.10f}')
print()
print("Dipoles (in local axes):")
print("{:>6}   {:<13} {:<13} {:<13}".format("Index", "x", "y", "z"))
for i, dipole in enumerate(dipoles_local):
    print(f'{i:>6}: {dipole[0]: 12.10f} {dipole[1]: 12.10f} {dipole[2]: 12.10f}')
print()
print("Quadrupoles (in local axes):")
print("{:>6}   {:<13} {:<13} {:<13} {:<13} {:<13} {:<13}".format("Index", "xx", "xy", "xz", "yy", "yz", "zz"))
for i, quadrupole in enumerate(quadrupoles_local):
    print(
        f'{i:>6}: {quadrupole[0,0]: 12.10f} {quadrupole[0,1]: 12.10f} {quadrupole[0,2]: 12.10f} {quadrupole[1,1]: 12.10f} {quadrupole[1,2]: 12.10f} {quadrupole[2,2]: 12.10f}'
    )
