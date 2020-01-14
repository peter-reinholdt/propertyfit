#!/usr/bin/env python

import sys
import pathlib
import warnings
import propertyfit
from propertyfit.structures import constraints

with open('results.csv', 'w') as csvfile:
    csvfile.write('resname,atomname,axis_type,axis_atomnames[0],axis_atomnames[1],charge,dipole[0],dipole[1],dipole[2],quadrupole[0],quadrupole[1],quadrupole[2],quadrupole[3],quadrupole[4],quadrupole[5]\n')
    for filename in sys.argv[1:]:
        path = pathlib.Path(filename)
        if not path.stat().st_size > 0:
            warnings.warn(f'Skipping empty file {filename}')
            continue
        section = ''
        charges = []
        dipoles = []
        quadrupoles = []
        stem = path.stem
        resname = stem.split("_")[1]
        restype = "_".join(stem.split("_")[2:4])
        name = '_'.join([resname, restype])
        if restype == "methyl_methyl":
            prefix = ''
            central_fragment = 1
        elif restype == "charged_methyl":
            prefix = 'N'
            central_fragment = 0
        elif restype == "neutral_methyl":
            prefix = 'n'
            central_fragment = 0
        elif restype == "methyl_charged":
            prefix = 'C'
            central_fragment = 1
        elif restype == "methyl_neutral":
            prefix = 'c'
            central_fragment = 1
        else:
            raise NotImplementedError(restype)
        con = constraints(f"constraints/{name}.constraints.new")
        with open(filename, 'r') as resultfile:
            for line in resultfile:
                if 'Charges' in line:
                    section = 'charges'
                    continue
                elif 'Dipoles' in line:
                    section = 'dipoles'
                    continue
                elif 'Quadrupoles' in line:
                    section = 'quadrupoles'
                    continue
                elif 'Index' in line:
                    continue
                
                if not line.split():
                    continue
                elif section == 'charges':
                    charges.append(line.split()[1])
                elif section == 'dipoles':
                    dipoles.append(line.split()[1:])
                elif section == 'quadrupoles':
                    quadrupoles.append(line.split()[1:])
        #@
        frag = con.fragments[central_fragment]
        for idx in frag.atomindices:
            atomname = frag.idx2atomname[idx]
            axis_type = frag.idx2axis_type[idx]
            axis_atomnames = frag.idx2axis_atomnames[idx]
            charge = charges[idx]
            dipole = dipoles[idx]
            quadrupole = quadrupoles[idx]
            csvfile.write(f'{prefix+resname},{atomname},{axis_type},{axis_atomnames[0]},{axis_atomnames[1]},{charge},{dipole[0]},{dipole[1]},{dipole[2]},{quadrupole[0]},{quadrupole[1]},{quadrupole[2]},{quadrupole[3]},{quadrupole[4]},{quadrupole[5]}\n')