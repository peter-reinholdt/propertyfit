#!/usr/bin/env python

import sys
import numpy as np
from propertyfit.structures import structure
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--qm-files',
                    "-qm",
                    dest="qm",
                    type=str,
                    nargs="+",
                    help="Get ESP from .fchk files (the ESP is evaluated with gaussian).")
parser.add_argument('--terachem-scrdirs',
                    dest="terachem_scrdirs",
                    type=str,
                    nargs='+',
                    help='List of scratch dirs from terachem, containing esp.xyz and jobname.geometry')
parser.add_argument('--orca-gbws', dest="orca_gbws", type=str, nargs='+', help='List of orca gbw files')
parser.add_argument('--orca-densities',
                    dest="orca_densities",
                    type=str,
                    nargs='+',
                    help='List of orca scfp/cisp/mdcip/... files')
parser.add_argument('--surface-rmin', dest='rmin', type=float, default=1.4, help='Set minimum vdW scale in bohr.')
parser.add_argument('--surface-rmax', dest='rmax', type=float, default=2.0, help='Set minimum vdW scale in bohr.')
parser.add_argument('--point-density',
                    dest='point_density',
                    type=float,
                    default=1.0,
                    help='Density of points on vdW surface')
parser.add_argument('--n-surfaces', dest='n_surfaces', type=int, default=2, help='Number of vdW surfaces to use')

args = parser.parse_args()
if not (args.qm or args.terachem_scrdirs or (args.orca_gbws and args.orca_densities)):
    raise ValueError("Incorrect specification of ESP source")

if args.qm:
    for fname in args.qm:
        s = structure()
        s.load_fchk(fname)
        s.compute_grid(rmin=args.rmin, rmax=args.rmax, pointdensity=args.point_density, nsurfaces=args.n_surfaces)
        s.compute_qm_esp_gaussian()
        s.save_h5(fname + ".h5")
if args.terachem_scrdirs:
    for folder_name in args.terachem_scrdirs:
        s = structure()
        s.load_esp_terachem(folder_name)
        s.save_h5(folder_name + ".h5")
if args.orca_gbws:
    assert len(args.orca_gbws) == len(args.orca_densities)
    for gbw, density in zip(args.orca_gbws, args.orca_densities):
        fname = "_".join([gbw, density])
        s = structure(vdw_grid_rmin=args.rmin,
                      vdw_grid_rmax=args.rmax,
                      vdw_grid_pointdensity=args.point_density,
                      vdw_grid_nsurfaces=args.n_surfaces)
        s.load_esp_orca(gbw, density)
        s.save_h5(fname + ".h5")
