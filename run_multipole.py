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

con.restraint = 0.0 # TODO add restraints for multipole
dipole_startguess = np.zeros(con.nparametersmu, dtype=np.float64)
quadrupole_startguess = np.zeros(con.nparameterstheta, dtype=np.float64)
parameters = np.hstack([con.q0, dipole_startguess, quadrupole_startguess])

fun = functools.partial(multipole_cost_function, structures=structures, constraints=con, weights=weights)
res = minimize(fun, x0=parameters, method=args.method, tol=1e-12, options={'maxiter': 1000})


print(res)
print()
print("=" * 85)
print()
print("Final result:")
print("Charges:")
print("{:>6}   {:<12}".format("Index", "Charge"))
for i, charge in enumerate(con.expand_charges(res.x[0:con.nparametersq])):
    print(f'{i:>6}: {charge: 12.10f}')
print()
print("Dipoles (in local axes):")
print("{:>6}   {:<13} {:<13} {:<13}".format("Index", "x", "y", "z"))
for i, dipole in enumerate(con.expand_dipoles(res.x[con.nparametersq:con.nparametersq +
                                                         con.nparametersmu]).reshape(-1, 3)):
    print(f'{i:>6}: {dipole[0]: 12.10f} {dipole[1]: 12.10f} {dipole[2]: 12.10f}')
print()
print("Quadrupoles:")
print("{:>6}   {:<13} {:<13} {:<13} {:<13} {:<13} {:<13}".format("Index", "xx", "xy", "xz", "yy", "yz", "zz"))
for i, quadrupole in enumerate(
        con.expand_quadrupoles(res.x[con.nparametersq + con.nparametersmu:con.nparametersq + con.nparametersmu +
                                          con.nparameterstheta]).reshape(-1, 3, 3)):
    print(
        f'{i:>6}: {quadrupole[0,0]: 12.10f} {quadrupole[0,1]: 12.10f} {quadrupole[0,2]: 12.10f} {quadrupole[1,1]: 12.10f} {quadrupole[1,2]: 12.10f} {quadrupole[2,2]: 12.10f}'
    )
