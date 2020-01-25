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

bohr2angstrom = 0.52917724900001
angstrom2bohr = 1.0 / (bohr2angstrom)
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


def memoize_on_first_arg_function(f):
    cache = {}

    @wraps(f)
    def wrapper(*args):
        if args[0] in cache:
            return cache[args[0]]
        else:
            cache[args[0]] = f(*args)
            return cache[args[0]]

    return wrapper


def memoize_on_first_arg_method(f):
    cache = {}

    @wraps(f)
    def wrapper(*args):
        # first arg on class method is self...
        # use args[1] instead
        if args[1] in cache:
            return cache[args[1]]
        else:
            cache[args[1]] = f(*args)
            return cache[args[1]]

    return wrapper
