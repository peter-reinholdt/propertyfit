#!/usr/bin/env python
"""
Utility routines and
conversion factors
"""

import os
import warnings
import glob
import numpy as np
import sh
import json
import contextlib
from numpy import nan
from functools import wraps

hartree2kjmol = 2625.5002
kjmol2hartree = 1.0 / hartree2kjmol


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


@contextlib.contextmanager
def return_to_cwd():
    cwd = os.getcwd()
    try:
        yield
    finally:
        os.chdir(cwd)


dipole_axis_nonzero = {}
polarizability_axis_nonzero = dict()
quadrupole_axis_nonzero = {}

# indexed by (axis_type, number_of_symmetric)
dipole_axis_nonzero[('internal_four_neighbors', (2, 1, 1))] = [False, True, True]
dipole_axis_nonzero[('internal_four_neighbors', (2, 2))] = [False, False, True]
dipole_axis_nonzero[('internal_four_neighbors', (1, 1, 1, 1))] = [False, False, True]
dipole_axis_nonzero[('internal_four_neighbors_symmetric', (3, 1))] = [False, False, True]
dipole_axis_nonzero[('internal_three_neighbors', (2, 1))] = [False, True, True]
dipole_axis_nonzero[('internal_three_neighbors', (3))] = [False, False, True]
dipole_axis_nonzero[('internal_three_neighbors', (1, 1, 1))] = [True, True, True]
dipole_axis_nonzero[('internal_two_neighbors', (2))] = [False, True, False]
dipole_axis_nonzero[('internal_two_neighbors', (1, 1))] = [True, True, False]
dipole_axis_nonzero[('terminal_three_adjacent_neighbors', (1, 2, 1))] = [False, True, True]
dipole_axis_nonzero[('terminal_three_adjacent_neighbors', (1, 3))] = [False, False, True]
dipole_axis_nonzero[('terminal_three_adjacent_neighbors', (1, 1, 1, 1))] = [True, True, True]
dipole_axis_nonzero[('terminal_two_adjacent_neighbors', (1, 2))] = [False, True, True]
dipole_axis_nonzero[('terminal_two_adjacent_neighbors', (1, 1, 1))] = [True, True, True]
dipole_axis_nonzero[('terminal_one_adjacent_neighbor', (1, 1))] = [True, False, True]
quadrupole_axis_nonzero[('internal_four_neighbors', (2, 1, 1))] = [False, False, True, True, True]
quadrupole_axis_nonzero[('internal_four_neighbors', (2, 2))] = [False, False, False, True, False]
quadrupole_axis_nonzero[('internal_four_neighbors', (1, 1, 1, 1))] = [True, True, True, True, True]
quadrupole_axis_nonzero[('internal_four_neighbors_symmetric', (3, 1))] = [False, False, False, False, True]
quadrupole_axis_nonzero[('internal_three_neighbors', (2, 1))] = [False, False, True, True, True]
quadrupole_axis_nonzero[('internal_three_neighbors', (3))] = [False, False, False, False, True]
quadrupole_axis_nonzero[('internal_three_neighbors', (1, 1, 1))] = [True, True, True, True, True]
quadrupole_axis_nonzero[('internal_two_neighbors', (2))] = [False, False, False, False, True]
quadrupole_axis_nonzero[('internal_two_neighbors', (1, 1))] = [True, False, False, True, True]
quadrupole_axis_nonzero[('terminal_three_adjacent_neighbors', (1, 2, 1))] = [False, False, True, True, True]
quadrupole_axis_nonzero[('terminal_three_adjacent_neighbors', (1, 3))] = [False, False, False, False, True]
quadrupole_axis_nonzero[('terminal_three_adjacent_neighbors', (1, 1, 1, 1))] = [True, True, True, True, True]
quadrupole_axis_nonzero[('terminal_two_adjacent_neighbors', (1, 2))] = [False, False, True, True, True]
quadrupole_axis_nonzero[('terminal_two_adjacent_neighbors', (1, 1, 1))] = [True, True, True, True, True]
quadrupole_axis_nonzero[('terminal_one_adjacent_neighbor', (1, 1))] = [False, True, False, True, True]
polarizability_axis_nonzero[('internal_four_neighbors', (2, 1, 1))] = [True] * 6
polarizability_axis_nonzero[('internal_four_neighbors', (2, 2))] = [True] * 6
polarizability_axis_nonzero[('internal_four_neighbors', (1, 1, 1, 1))] = [True] * 6
polarizability_axis_nonzero[('internal_four_neighbors_symmetric', (3, 1))] = [True] * 6
polarizability_axis_nonzero[('internal_three_neighbors', (2, 1))] = [True] * 6
polarizability_axis_nonzero[('internal_three_neighbors', (3))] = [True] * 6
polarizability_axis_nonzero[('internal_three_neighbors', (1, 1, 1))] = [True] * 6
polarizability_axis_nonzero[('internal_two_neighbors', (2))] = [True] * 6
polarizability_axis_nonzero[('internal_two_neighbors', (1, 1))] = [True] * 6
polarizability_axis_nonzero[('terminal_three_adjacent_neighbors', (1, 2, 1))] = [True] * 6
polarizability_axis_nonzero[('terminal_three_adjacent_neighbors', (1, 3))] = [True] * 6
polarizability_axis_nonzero[('terminal_three_adjacent_neighbors', (1, 1, 1, 1))] = [True] * 6
polarizability_axis_nonzero[('terminal_two_adjacent_neighbors', (1, 2))] = [True] * 6
polarizability_axis_nonzero[('terminal_two_adjacent_neighbors', (1, 1, 1))] = [True] * 6
polarizability_axis_nonzero[('terminal_one_adjacent_neighbor', (1, 1))] = [True] * 6

dipole_axis_nonzero[('zthenx', (2, 1, 1))] = [True, False, True]  #  (198, 155, 198)
dipole_axis_nonzero[('zthenx', (2, 2))] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('zthenx', (1, 1, 1, 1))] = [False, False, True]  #  (612, 518, 660)
dipole_axis_nonzero[('zthenx', (3, 1))] = [False, False, True]  #  (154, 232, 242)
dipole_axis_nonzero[('zthenx', (2, 1))] = [True, False, True]  #  (51, 36, 71)
dipole_axis_nonzero[('zthenx', 3)] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('zthenx', (1, 1, 1))] = [True, False, True]  #  (615, 863, 1145)
dipole_axis_nonzero[('zthenx', 2)] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('zthenx', (1, 1))] = [True, False, True]  #  (82, 113, 139)
dipole_axis_nonzero[('zthenx', (1, 2, 1))] = [False, False, True]  #  (740, 740, 744)
dipole_axis_nonzero[('zthenx', (1, 3))] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('zthenx', (1, 1, 1, 1))] = [False, False, True]  #  (612, 518, 660)
dipole_axis_nonzero[('zthenx', (1, 2))] = [False, False, True]  #  (5, 5, 5)
dipole_axis_nonzero[('zthenx', (1, 1, 1))] = [True, False, True]  #  (615, 863, 1145)
dipole_axis_nonzero[('zthenx', (1, 1))] = [True, False, True]  #  (82, 113, 139)
dipole_axis_nonzero[('bisector', (2, 1, 1))] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('bisector', (2, 2))] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('bisector', (1, 1, 1, 1))] = [True, True, False]  #  (2, 2, 2)
dipole_axis_nonzero[('bisector', (3, 1))] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('bisector', (2, 1))] = [False, True, True]  #  (42, 26, 37)
dipole_axis_nonzero[('bisector', 3)] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('bisector', (1, 1, 1))] = [False, True, True]  #  (36, 23, 37)
dipole_axis_nonzero[('bisector', 2)] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('bisector', (1, 1))] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('bisector', (1, 2, 1))] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('bisector', (1, 3))] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('bisector', (1, 1, 1, 1))] = [True, True, False]  #  (2, 2, 2)
dipole_axis_nonzero[('bisector', (1, 2))] = [False, False, False]  #  (0, 0, 0)
dipole_axis_nonzero[('bisector', (1, 1, 1))] = [False, True, True]  #  (36, 23, 37)
dipole_axis_nonzero[('bisector', (1, 1))] = [False, False, False]  #  (0, 0, 0)
quadrupole_axis_nonzero[('zthenx', (2, 1, 1))] = [True, False, True, True, False]  #  (188, 149, 161, 201, 124)
quadrupole_axis_nonzero[('zthenx', (2, 2))] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('zthenx', (1, 1, 1, 1))] = [True, False, False, True, False]  #  (657, 507, 493, 608, 383)
quadrupole_axis_nonzero[('zthenx', (3, 1))] = [True, False, False, True, False]  #  (241, 211, 134, 243, 204)
quadrupole_axis_nonzero[('zthenx', (2, 1))] = [True, False, True, True, True]  #  (71, 38, 51, 71, 41)
quadrupole_axis_nonzero[('zthenx', 3)] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('zthenx', (1, 1, 1))] = [True, False, True, True, False]  #  (1159, 898, 580, 1160, 961)
quadrupole_axis_nonzero[('zthenx', 2)] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('zthenx', (1, 1))] = [True, False, True, True, False]  #  (134, 127, 87, 139, 132)
quadrupole_axis_nonzero[('zthenx', (1, 2, 1))] = [True, False, False, True, False]  #  (744, 735, 577, 744, 738)
quadrupole_axis_nonzero[('zthenx', (1, 3))] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('zthenx', (1, 1, 1, 1))] = [True, False, False, True, False]  #  (657, 507, 493, 608, 383)
quadrupole_axis_nonzero[('zthenx', (1, 2))] = [True, False, False, True, False]  #  (5, 5, 5, 5, 5)
quadrupole_axis_nonzero[('zthenx', (1, 1, 1))] = [True, False, True, True, False]  #  (1159, 898, 580, 1160, 961)
quadrupole_axis_nonzero[('zthenx', (1, 1))] = [True, False, True, True, False]  #  (134, 127, 87, 139, 132)
quadrupole_axis_nonzero[('bisector', (2, 1, 1))] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('bisector', (2, 2))] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('bisector', (1, 1, 1, 1))] = [True, True, True, True, True]  #  (2, 2, 2, 2, 2)
quadrupole_axis_nonzero[('bisector', (3, 1))] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('bisector', (2, 1))] = [True, False, False, True, False]  #  (42, 42, 27, 42, 34)
quadrupole_axis_nonzero[('bisector', 3)] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('bisector', (1, 1, 1))] = [True, False, True, True, False]  #  (37, 36, 37, 37, 19)
quadrupole_axis_nonzero[('bisector', 2)] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('bisector', (1, 1))] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('bisector', (1, 2, 1))] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('bisector', (1, 3))] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('bisector', (1, 1, 1, 1))] = [True, True, True, True, True]  #  (2, 2, 2, 2, 2)
quadrupole_axis_nonzero[('bisector', (1, 2))] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
quadrupole_axis_nonzero[('bisector', (1, 1, 1))] = [True, False, True, True, False]  #  (37, 36, 37, 37, 19)
quadrupole_axis_nonzero[('bisector', (1, 1))] = [False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('zthenx', (2, 1, 1))] = [True, False, True, True, False, True]  #  (201, 154, 192, 201, 111)
polarizability_axis_nonzero[('zthenx', (2, 2))] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('zthenx', (1, 1, 1, 1))] = [True, False, False, True, False, True]  #  (657, 521, 574, 652, 496)
polarizability_axis_nonzero[('zthenx', (3, 1))] = [True, False, True, True, False, True]  #  (243, 236, 220, 243, 169)
polarizability_axis_nonzero[('zthenx', (2, 1))] = [True, False, True, True, True, True]  #  (71, 48, 36, 71, 43)
polarizability_axis_nonzero[('zthenx', 3)] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('zthenx', (1, 1, 1))] = [True, False, True, True, False, True]  #  (1155, 954, 628, 1159, 893)
polarizability_axis_nonzero[('zthenx', 2)] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('zthenx', (1, 1))] = [True, False, True, True, False, True]  #  (139, 124, 115, 139, 116)
polarizability_axis_nonzero[('zthenx', (1, 2, 1))] = [True, False, True, True, False, True]  #  (744, 743, 645, 682, 731)
polarizability_axis_nonzero[('zthenx', (1, 3))] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('zthenx', (1, 1, 1, 1))] = [True, False, False, True, False, True]  #  (657, 521, 574, 652, 496)
polarizability_axis_nonzero[('zthenx', (1, 2))] = [True, False, False, True, False, True]  #  (5, 5, 5, 5, 5)
polarizability_axis_nonzero[('zthenx', (1, 1, 1))] = [True, False, True, True, False, True]  #  (1155, 954, 628, 1159, 893)
polarizability_axis_nonzero[('zthenx', (1, 1))] = [True, False, True, True, False, True]  #  (139, 124, 115, 139, 116)
polarizability_axis_nonzero[('bisector', (2, 1, 1))] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('bisector', (2, 2))] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('bisector', (1, 1, 1, 1))] = [True, True, True, True, True, True]  #  (2, 2, 2, 2, 2)
polarizability_axis_nonzero[('bisector', (3, 1))] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('bisector', (2, 1))] = [True, False, True, True, False, True]  #  (42, 40, 41, 42, 29)
polarizability_axis_nonzero[('bisector', 3)] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('bisector', (1, 1, 1))] = [True, False, True, True, False, True]  #  (37, 35, 35, 37, 31)
polarizability_axis_nonzero[('bisector', 2)] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('bisector', (1, 1))] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('bisector', (1, 2, 1))] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('bisector', (1, 3))] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('bisector', (1, 1, 1, 1))] = [True, True, True, True, True, True]  #  (2, 2, 2, 2, 2)
polarizability_axis_nonzero[('bisector', (1, 2))] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
polarizability_axis_nonzero[('bisector', (1, 1, 1))] = [True, False, True, True, False, True]  #  (37, 35, 35, 37, 31)
polarizability_axis_nonzero[('bisector', (1, 1))] = [False, False, False, False, False, False]  #  (0, 0, 0, 0, 0)
