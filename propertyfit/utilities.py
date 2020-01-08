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
    warnings.warn("Running without support for horton", RuntimeWarning)
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


def memoize_on_first_arg(f):
    cache = {}

    @wraps(f)
    def wrapper(*args):
        if args[0] in cache:
            return cache[args[0]]
        else:
            cache[args[0]] = f(*args)
            return cache[args[0]]

    return wrapper


name2number = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn":
    25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd":
    48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59,
    "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb":
    82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93,
    "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104,
    "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114,
    "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}

number2name = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25:
    "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36:
    "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag",
    48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59:
    "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70:
    "Yb", 71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl",
    82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U", 93:
    "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm", 101: "Md", 102: "No", 103: "Lr", 104:
    "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds", 111: "Rg", 112: "Cn", 113: "Nh", 114:
    "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
}

vdw_radii = {
    1: 2.26767118629, 2: 2.64561638401, 3: 3.43930129921, 4: 2.89128076253, 5: 3.62827389807, 6: 3.21253418058, 7:
    2.9290752823, 8: 2.87238350264, 9: 2.77789720321, 10: 2.91017802241, 11: 4.28967799407, 12: 3.26922596024, 13:
    3.47709581899, 14: 3.96842457602, 15: 3.40150677944, 16: 3.40150677944, 17: 3.30702048001, 18: 3.55268485853, 19:
    5.19674646859, 20: 4.36526703362, 21: 3.9873218359, 22: nan, 23: nan, 24: nan, 25: nan, 26: nan, 27: nan, 28:
    3.08025336138, 29: 2.64561638401, 30: 2.62671912412, 31: 3.53378759864, 32: 3.9873218359, 33: 3.49599307887, 34:
    3.5904793783, 35: 3.49599307887, 36: 3.81724649693, 37: 5.72586974539, 38: 4.70541771156, 39: nan, 40: nan, 41:
    nan, 42: nan, 43: nan, 44: nan, 45: nan, 46: 3.08025336138, 47: 3.25032870036, 48: 2.98576706195, 49:
    3.64717115796, 50: 4.10070539522, 51: 3.89283553647, 52: 3.89283553647, 53: 3.74165745739, 54: 4.08180813533, 55:
    6.48176014083, 56: 5.06446564939, 57: nan, 58: nan, 59: nan, 60: nan, 61: nan, 62: nan, 63: nan, 64: nan, 65: nan,
    66: nan, 67: nan, 68: nan, 69: nan, 70: nan, 71: nan, 72: nan, 73: nan, 74: nan, 75: nan, 76: nan, 77: nan, 78:
    3.30702048001, 79: 3.13694514104, 80: 2.9290752823, 81: 3.70386293761, 82: 3.81724649693, 83: 3.91173279636, 84:
    3.7227601975, 85: 3.81724649693, 86: 4.15739717487, 87: 6.57624644025, 88: 5.34792454768, 89: nan, 90: nan, 91:
    nan, 92: 3.51489033876, 93: nan, 94: nan, 95: nan, 96: nan, 97: nan, 98: nan, 99: nan, 100: nan, 101: nan, 102:
    nan, 103: nan, 104: nan, 105: nan, 106: nan, 107: nan, 108: nan, 109: nan, 110: nan, 111: nan, 112: nan, 113: nan,
    114: nan, 115: nan, 116: nan, 117: nan, 118: nan
}
