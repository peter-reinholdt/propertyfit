#!/usr/bin/env python


import sys
import numpy as np
import functools
from scipy.optimize import minimize
from propertyfit.structures import structure, constraints
from propertyfit.costfunctions import charge_cost_function
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--qm-files', "-qm", dest="qm", type=str, nargs="+", help="Specify a number of .molden or .fchk files")
parser.add_argument('--h5-files', "-h5", dest="h5", type=str, nargs="+", help="Specify a number of .h5 files with pre-computed potentials. If this is used, the\
        vdW surface specification arguments are ignored, since the .h5 file already contains this information")
parser.add_argument('--terachem-scrdirs', dest="terachem_scrdirs", type=str, nargs='+', help='List of scratch dirs from terachem, containing esp.xyz and jobname.geometry')
parser.add_argument('--orca-gbws', dest="orca_gbws", type=str, nargs='+', help='List of orca gbw files')
parser.add_argument('--orca-densities', dest="orca_densities", type=str, nargs='+', help='List of orca scfp/cisp/mdcip/... files')
parser.add_argument('--h5-file-list', dest='h5_filelist', type=str, help='Read which h5 files to use from a file')
parser.add_argument('--surface-rmin', dest='rmin', type=float, default=1.4, help='Set minimum vdW scale in bohr.')
parser.add_argument('--surface-rmax', dest='rmax', type=float, default=2.0, help='Set minimum vdW scale in bohr.')
parser.add_argument('--point-density', dest='point_density', type=float, default=1.0, help='Density of points on vdW surface')
parser.add_argument('--topology', dest='top', type=str, help='Provide file for information about symmetry-equivalent atoms and more.', required=True)
parser.add_argument('--n-surfaces', dest='n_surfaces', type=int, default=2, help='Number of vdW surfaces to use')
parser.add_argument('--restraint', dest='restraint', type=float, default=0.0, help='Strength of harmonic restraint towards charges from topology file')
parser.add_argument('--method', dest='method', default='slsqp', help='Which optimizer to use')
parser.add_argument('-o', dest='output', type=str, default='charges.dat', help='Name of file to write charges to')

args = parser.parse_args()
if not (args.qm or args.h5 or args.h5_filelist or args.terachem_scrdirs or (args.orca_gbws and args.orca_densities)):
    raise ValueError("Please specify either a set of .h5 files or a set of qm files")


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
if args.qm:
    for fname in args.qm:
        s = structure()
        s.load_qm(fname, np.array([0.0, 0.0, 0.0]))
        s.compute_grid(rmin=args.rmin, rmax=args.rmax, pointdensity=args.point_density, nsurfaces=args.n_surfaces)
        s.compute_rinvmat()
        s.compute_xyzmat()
        s.compute_qm_esp()
        s.save_h5(fname + ".h5")
        structures.append(s)
if args.terachem_scrdirs:
    for folder_name in args.terachem_scrdirs:
        s = structure()
        s.load_esp_terachem(folder_name)
        s.compute_rinvmat()
        s.compute_xyzmat()
        structures.append(s)
if args.orca_gbws:
    assert len(args.orca_gbws) == len(args.orca_densities)
    for gbw, density in zip(args.orca_gbws, args.orca_densities):
        fname = "_".join([gbw, density])
        s = structure(vdw_grid_rmin=args.rmin, vdw_grid_rmax=args.rmax, vdw_grid_pointdensity=args.point_density, vdw_grid_nsurfaces=args.n_surfaces)
        s.load_esp_orca(gbw, density)
        s.save_h5(fname + ".h5")
        structures.append(s)
        


        

#use partial to wrap cost function, so we only need a single qtest argument (and not constraints, structures)
#then we can call fun(qtest) instead of charge_cost_function(qtest, structures, constraints)

con.restraint = args.restraint
q0 = con.q0


fun = functools.partial(charge_cost_function, structures=structures, constraints=con)
res = minimize(fun, x0=q0, method=args.method)

print(res)
print("\n========================================================\n")
print("Final result:")
with open(args.output, "w") as f:
    for q in con.expand_q(res.x):
        print(q)
        f.write("{}\n".format(q))
