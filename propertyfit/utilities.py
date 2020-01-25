#!/usr/bin/env python
"""
Utility routines and
conversion factors
"""

import os
import warnings
try:
    import horton
except:
    pass
    #warnings.warn("Running without support for horton", RuntimeWarning)
import glob
import numpy as np
import sh
import json
from numpy import nan
from functools import wraps

hartree2kjmol = 2625.5002
kjmol2hartree = 1.0 / hartree2kjmol


def load_qmfiles(regex):
    from structures import structure  #sorry!
    files = glob.glob(regex)
    structures = []
    for i in files:
        io = horton.IOData.from_file(i)
        try:
            grepfield = sh.grep("E-field", i, "-A1")
            field = np.array([float(x) for x in grepfield.stdout.split()[6:9]])
        except:
            print("INFO: no field information found in {}. Assuming zero field.".format(i))
            field = np.array([0., 0., 0.])

        s = structure()
        s.load_qm(i, field)
        structures.append(s)
    return structures


def load_json(filename):
    with open(filename, "r") as f:
        res = json.load(f)
    return res


def load_geometry_from_molden(filename):
    with open(filename, 'r') as f:
        coordinates = []
        elements = []
        for line in f:
            if '[Atoms]' in line:
                unit = line.split()[-1]
                break
        for line in f:
            if '[' in line:
                break
            else:
                element, index, charge, x, y, z = line.split()
                coordinates.append([float(c) for c in [x, y, z]])
                elements.append(element)
    #qdata should be in bohr for the coords
    coordinates = np.array(coordinates)
    if unit.lower() == 'angs':
        return coordinates * 1.8897259886, elements
    else:
        return coordinates, elements

dipole_axis_nonzero = {}
dipole_axis_nonzero[('internal_four_neighbors', (2, 1, 1))] = [False, True, True] 
dipole_axis_nonzero[('internal_four_neighbors', (2, 2)]     = [False, False, True] 
dipole_axis_nonzero[('internal_four_neighbors', (1, 1, 1, 1)] = [False, False, True] 
dipole_axis_nonzero[('internal_four_neighbors_symmetric', (3, 1)] = [False, False, True] 
dipole_axis_nonzero[('internal_three_neighbors', (2,1)] = [False, True, True] 
dipole_axis_nonzero[('internal_three_neighbors', (3)] = [False, False, True] 
dipole_axis_nonzero[('internal_three_neighbors', (1, 1, 1)] = [True, True, True] 
dipole_axis_nonzero[('internal_two_neighbors', (2)] = [False, True, False] 
dipole_axis_nonzero[('internal_two_neighbors', (1,1)] = [True, True, False] 
dipole_axis_nonzero[('terminal_three_adjacent_neighbors', (1,2,1)] = [False, True, True]
dipole_axis_nonzero[('terminal_three_adjacent_neighbors', (1,3)] = [False, False, True]
dipole_axis_nonzero[('terminal_three_adjacent_neighbors', (1,1,1,1)] = [True, True, True]
dipole_axis_nonzero[('terminal_two_adjacent_neighbors', (1,2)] = [False, True, True]
dipole_axis_nonzero[('terminal_two_adjacent_neighbors', (1,1,1)] = [True, True, True]
dipole_axis_nonzero[('terminal_one_adjacent_neighbor', (1,1)] = [True, False, True]
quadrupole_axis_nonzero = {}
quadrupole_axis_nonzero[('internal_four_neighbors', (2, 1, 1))] = [False, False, True, True, True]
quadrupole_axis_nonzero[('internal_four_neighbors', (2, 2)]     = [False, False, False, True, False]
quadrupole_axis_nonzero[('internal_four_neighbors', (1, 1, 1, 1)] = [True, True, True, True, True]
quadrupole_axis_nonzero[('internal_four_neighbors_symmetric', (3, 1)] = [False, False, False, False, True]
quadrupole_axis_nonzero[('internal_three_neighbors', (2,1)] = [False, False, True, True, True]
quadrupole_axis_nonzero[('internal_three_neighbors', (3)] = [False, False, False, False, True]
quadrupole_axis_nonzero[('internal_three_neighbors', (1, 1, 1)] = [True, True, True, True, True]
quadrupole_axis_nonzero[('internal_two_neighbors', (2)] = [False, False, False, False, True]
quadrupole_axis_nonzero[('internal_two_neighbors', (1,1)] = [True, False, False, True, True]
quadrupole_axis_nonzero[('terminal_three_adjacent_neighbors', (1,2,1)] = [False, False, True, True, True]
quadrupole_axis_nonzero[('terminal_three_adjacent_neighbors', (1,3)] = [False, False, False, False, True]
quadrupole_axis_nonzero[('terminal_three_adjacent_neighbors', (1,1,1,1)] = [True, True, True, True, True]
quadrupole_axis_nonzero[('terminal_two_adjacent_neighbors', (1,2)] = [False, False, True, True, True]
quadrupole_axis_nonzero[('terminal_two_adjacent_neighbors', (1,1,1)] = [True, True, True, True, True]
quadrupole_axis_nonzero[('terminal_one_adjacent_neighbor', (1,1)] = [False, True, False, True, True]
