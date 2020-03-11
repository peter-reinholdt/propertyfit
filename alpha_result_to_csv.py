#!/usr/bin/env python

import re
import sys
import pathlib
import warnings
import propertyfit
from propertyfit.structures import constraints

with open('alpha_results.csv', 'w') as csvfile:
    csvfile.write(
        'resname,atomname,axis_type,axis_atomnames,polarizability[0],polarizability[1],polarizability[2],polarizability[3],polarizability[4],polarizability[5]\n'
    )
    for filename in sys.argv[2:]:
        path = pathlib.Path(filename)
        if not path.stat().st_size > 0:
            warnings.warn(f'Skipping empty file {filename}')
            continue
        section = ''
        polarizabilities = []
        stem = path.stem
        print(stem)
        resname = re.search('[A-Z][A-Z][A-Z]', stem).group(0)
        restype = re.search(f'{resname}_([a-z]+_[a-z]+)', stem).group(1)
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
        con = constraints(f"{sys.argv[1]}/{name}.constraints.new")
        with open(filename, 'r') as resultfile:
            for line in resultfile:
                if 'Polarizabilities' in line:
                    section = 'polarizabilities'
                    continue
                elif 'Index' in line:
                    continue
                elif section == 'polarizabilities':
                    polarizabilities.append(line.split()[1:])
                elif 'Change' in line:
                    break
        frag = con.fragments[central_fragment]
        for idx in frag.atomindices:
            atomname = con.atomnames[idx]
            axis_type = con.axis_types[idx]
            axis_atomnames = con.axis_atomnames[idx]
            polarizability = polarizabilities[idx]
            csvfile.write(
                f'{prefix+resname},{atomname},{axis_type},{" ".join(axis_atomnames)},{polarizability[0]},{polarizability[1]},{polarizability[2]},{polarizability[3]},{polarizability[4]},{polarizability[5]}\n'
            )
