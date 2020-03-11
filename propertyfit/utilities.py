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
dipole_axis_nonzero[('zthenx', (1, 1, 1, 1))] = [False, True, True]  #  (96, 135, 138)
dipole_axis_nonzero[('zthenx', (3, 1))] = [False, False, True]  #  (154, 232, 242)
dipole_axis_nonzero[('zthenx', (2, 1))] = [True, False, True]  #  (51, 36, 71)
dipole_axis_nonzero[('zthenx', (1, 1, 1))] = [True, False, True]  #  (484, 283, 475)
dipole_axis_nonzero[('zthenx', (1, 1))] = [True, False, True]  #  (82, 56, 82)
dipole_axis_nonzero[('zthenx', (-1, 2, 1))] = [False, False, True]  #  (740, 740, 744)
dipole_axis_nonzero[('zthenx', (-1, 1, 1, 1))] = [False, False, True]  #  (516, 515, 522)
dipole_axis_nonzero[('zthenx', (-1, 2))] = [False, False, True]  #  (5, 5, 5)
dipole_axis_nonzero[('zthenx', (-1, 1, 1))] = [False, False, True]  #  (539, 580, 670)
dipole_axis_nonzero[('zthenx', (-1, 1))] = [False, False, True]  #  (57, 57, 57)
dipole_axis_nonzero[('bisector', (1, 1, 1, 1))] = [True, True, False]  #  (2, 2, 2)
dipole_axis_nonzero[('bisector', (2, 1))] = [False, True, True]  #  (42, 26, 37)
dipole_axis_nonzero[('bisector', (1, 1, 1))] = [False, True, True]  #  (36, 23, 37)
quadrupole_axis_nonzero[('zthenx', (2, 1, 1))] = [True, False, True, True, False]  #  (180, 183, 128, 201, 167)
quadrupole_axis_nonzero[('zthenx', (1, 1, 1, 1))] = [True, True, False, False, True]  #  (134, 137, 95, 86, 90)
quadrupole_axis_nonzero[('zthenx', (3, 1))] = [True, False, False, True, False]  #  (214, 236, 231, 243, 235)
quadrupole_axis_nonzero[('zthenx', (2, 1))] = [True, False, True, True, True]  #  (71, 40, 51, 71, 39)
quadrupole_axis_nonzero[('zthenx', (1, 1, 1))] = [True, False, True, True, False]  #  (490, 329, 283, 490, 388)
quadrupole_axis_nonzero[('zthenx', (1, 1))] = [True, False, True, True, False]  #  (52, 78, 82, 82, 80)
quadrupole_axis_nonzero[('zthenx', (-1, 2, 1))] = [True, False, False, True, False]  #  (744, 738, 722, 744, 739)
quadrupole_axis_nonzero[('zthenx', (-1, 1, 1, 1))] = [True, False, False, True, False]  #  (522, 522, 517, 522, 496)
quadrupole_axis_nonzero[('zthenx', (-1, 2))] = [True, False, False, True, False]  #  (5, 5, 5, 5, 5)
quadrupole_axis_nonzero[('zthenx', (-1, 1, 1))] = [True, False, False, True, False]  #  (669, 667, 548, 670, 670)
quadrupole_axis_nonzero[('zthenx', (-1, 1))] = [True, False, False, True, False]  #  (57, 57, 57, 57, 57)
quadrupole_axis_nonzero[('bisector', (1, 1, 1, 1))] = [True, True, True, True, True]  #  (2, 2, 2, 2, 2)
quadrupole_axis_nonzero[('bisector', (2, 1))] = [True, False, False, True, False]  #  (42, 42, 39, 42, 41)
quadrupole_axis_nonzero[('bisector', (1, 1, 1))] = [True, False, True, True, False]  #  (37, 37, 37, 37, 35)
polarizability_axis_nonzero[('zthenx', (2, 1, 1))] = [True, False, True, True, False, True]  #  (201, 154, 192, 201, 111)
polarizability_axis_nonzero[('zthenx', (1, 1, 1, 1))] = [True, True, False, True, True, True]  #  (138, 138, 73, 138, 130)
polarizability_axis_nonzero[('zthenx', (3, 1))] = [True, False, True, True, False, True]  #  (243, 236, 220, 243, 169)
polarizability_axis_nonzero[('zthenx', (2, 1))] = [True, False, True, True, True, True]  #  (71, 48, 36, 71, 43)
polarizability_axis_nonzero[('zthenx', (1, 1, 1))] = [True, False, True, True, False, True]  #  (490, 286, 381, 490, 252)
polarizability_axis_nonzero[('zthenx', (1, 1))] = [True, False, True, True, False, True]  #  (82, 67, 78, 82, 59)
polarizability_axis_nonzero[('zthenx', (-1, 2, 1))] = [True, False, True, True, False, True]  #  (744, 743, 645, 682, 731)
polarizability_axis_nonzero[('zthenx', (-1, 1, 1, 1))] = [True, False, False, True, False, True]  #  (519, 521, 501, 514, 488)
polarizability_axis_nonzero[('zthenx', (-1, 2))] = [True, False, False, True, False, True]  #  (5, 5, 5, 5, 5)
polarizability_axis_nonzero[('zthenx', (-1, 1, 1))] = [True, False, False, True, False, True]  #  (665, 668, 423, 669, 641)
polarizability_axis_nonzero[('zthenx', (-1, 1))] = [True, False, True, True, False, True]  #  (57, 57, 37, 57, 57)
polarizability_axis_nonzero[('bisector', (1, 1, 1, 1))] = [True, True, True, True, True, True]  #  (2, 2, 2, 2, 2)
polarizability_axis_nonzero[('bisector', (2, 1))] = [True, False, True, True, False, True]  #  (42, 40, 41, 42, 29)
polarizability_axis_nonzero[('bisector', (1, 1, 1))] = [True, False, True, True, False, True]  #  (37, 35, 35, 37, 31)
