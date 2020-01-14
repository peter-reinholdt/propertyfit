#!/usr/bin/env python

# standalone script to assign CP3 anisotropic parameters to a pdb

import sys
import numpy as np
import pandas
import pyframe

def bisector(point1, point2, point3):
    """
    decide local frame unit vectors x, y, z
    point1: coordinates of the atom in the origin of the local coordinate system
    point2: coordinates of first atom to define bisector
    point3: coordinates of second atom to define bisector
    """
    #create z-axis
    v1 = point2 - point1
    v1 = v1 / np.linalg.norm(v1)
    v2 = point3 - point1
    v2 = v2 / np.linalg.norm(v2)
    z = v1 + v2
    z = z / np.linalg.norm(z)

    #create x-axis by rejection
    x = point3 - point1
    x = x - (np.dot(x, z) / np.dot(z, z)) * z
    x = x / np.sqrt(np.sum(x * x))

    #dot = np.dot(v2, z)
    #x = v2 - dot * z
    #x = x / np.linalg.norm(x)

    #right hand rule for y
    y = np.cross(z, x)
    return np.vstack([x, y, z]).T


def zthenx(point1, point2, point3):
    """
    decide local frame unit vectors x, y, z
    point1: coordinates of the atom in the origin of the local coordinate system
    point2: coordinates of the atom to which the local z-axis is created
    point3: coordinates of a third atom, with which the local x-axis is created 
    """

    #create z-axis
    z = point2 - point1
    z = z / np.sqrt(np.sum(z * z))

    #create x-axis by rejection
    x = point3 - point1
    x = x - (np.dot(x, z) / np.dot(z, z)) * z
    x = x / np.sqrt(np.sum(x * x))

    #right hand rule for y
    y = np.cross(z, x)

    #rotation matrix
    return np.vstack([x, y, z]).T

parameters = pandas.read_csv("results.csv")
system = pyframe.MolecularSystem(sys.argv[1])
system.add_region('all', fragments=system.fragments, use_standard_potentials=True, standard_potential_model='cp3')
project = pyframe.Project()
project.scratch_dir = '/tmp'
project.create_embedding_potential(system)
atoms = []
for fragment in system.fragments.values():
    atoms += fragment.atoms

charges = []
dipoles = []
quadrupoles = []
rotation_matrices = []

equivalent_names = {'H': ['H1'],
                     'H1': ['H'],
                    'HA1': ['HA2', 'HA3'], #HA1, HA2, HA3 equivalent
                    'HA2': ['HA1', 'HA2'],
                    'HA3': ['HA1', 'HA2'],
                    'HB1': ['HB2', 'HB3'], #HB1, HB2, HB3 equivalent
                    'HB2': ['HB1', 'HB2'],
                    'HB3': ['HB1', 'HB2'],
                    'HG1': ['HG2', 'HG3'], #HG1, HG2, HG3 equivalent
                    'HG2': ['HG1', 'HG2'],
                    'HG3': ['HG1', 'HG2'],
                    'HD1': ['HD2', 'HD3'], #HD1, HD2, HD3 equivalent
                    'HD2': ['HD1', 'HD2'],
                    'HD3': ['HD1', 'HD2'],
                    'HE1': ['HE2', 'HE3'], #HE1, HE2, HE3 equivalent
                    'HE2': ['HE1', 'HE2'],
                    'HE3': ['HE1', 'HE2'],
                    'HG11': ['HG12', 'HG13'], #HG11, HG12, HG13 equivalent
                    'HG12': ['HG11', 'HG12'],
                    'HG13': ['HG11', 'HG12'],
                    'HG21': ['HG22', 'HG23'], #HG21, HG22, HG23 equivalent
                    'HG22': ['HG21', 'HG22'],
                    'HG23': ['HG21', 'HG22'],}

for fragment in system.fragments.values():
    residue = parameters[parameters.resname == fragment.name]

    fragment_coordinates = [atom.coordinate for atom in fragment.atoms]
    fragment_atomnames = np.array([atom.name for atom in fragment.atoms])

    for iatom, atom in enumerate(fragment.atoms):
        if atom.name in residue.atomname.values:
            p = residue[residue.atomname == atom.name]
        else:
            for name in equivalent_names[atom.name]:
                if name in residue.atomname.values:
                    p = residue[residue.atomname == name]
                    break

        charges.append(p.charge.values[0])
        dipoles.append([p["dipole[0]"].values[0], p["dipole[1]"].values[0], p["dipole[2]"].values[0]])
        quadrupoles.append([p["quadrupole[0]"].values[0], p["quadrupole[1]"].values[0], p["quadrupole[2]"].values[0], p["quadrupole[3]"].values[0], p["quadrupole[4]"].values[0], p["quadrupole[5]"].values[0]])

        point1 = atom.coordinate
        point1_idx = iatom
        

        # get other points
        point2_atomnames = list(p["axis_atomnames[0]"].values)
        extra_names = []
        for name in point2_atomnames:
            if name in equivalent_names:
                extra_names += equivalent_names[name]
        all_names = np.array([list(set(point2_atomnames + extra_names))])
        locs = np.where(all_names.T == fragment_atomnames)
        point2_idx = list(set(locs[1][:]).difference(set([point1_idx])))[0]
        point2 = fragment_coordinates[point2_idx]

        point3_atomnames = list(p["axis_atomnames[1]"].values)
        extra_names = []
        for name in point3_atomnames:
            if name in equivalent_names:
                extra_names += equivalent_names[name]
        all_names = np.array([list(set(point3_atomnames + extra_names))])
        locs = np.where(all_names.T == fragment_atomnames)
        point3_idx = list(set(locs[1][:]).difference(set([point1_idx, point2_idx])))[0]
        point3 = fragment_coordinates[point3_idx]


        if p.axis_type.values[0] == "zthenx":
            R = zthenx(point1, point2, point3)
            rotation_matrices.append(R)
        elif p.axis_type.values[0] == "bisector":
            R = bisector(point1, point2, point3)
            rotation_matrices.append(R)
        else:
            raise NotImplementedError

print('@COORDINATES')
print(len(atoms))
print('AA')
for atom in atoms:
    print('{:6s} {:16.12f} {:16.12f} {:16.12f} {:12d} ! {}'.format(atom.element, *atom.coordinate, atom.number, atom.name))

print('@MULTIPOLES')
print('ORDER 0')
print(len(atoms))
for atom, charge in zip(atoms, charges):
    print('{:<12d} {:16.12f}'.format(atom.number, charge))

# v' = R @ v
# M' = R @ M @ R.T

print('ORDER 1')
print(len(atoms))
for idx, atom in enumerate(atoms):
    dipole = rotation_matrices[idx] @ dipoles[idx]
    print('{:<12d} {:16.12f} {:16.12f} {:16.12f}'.format(atom.number, *dipole))

print('ORDER 2')
print(len(atoms))
for idx, atom in enumerate(atoms):
    quadrupole = np.zeros((3,3))
    quadrupole[0,0] = quadrupoles[idx][0]
    quadrupole[0,1] = quadrupoles[idx][1]
    quadrupole[1,0] = quadrupoles[idx][1]
    quadrupole[0,2] = quadrupoles[idx][2]
    quadrupole[2,0] = quadrupoles[idx][2]
    quadrupole[1,1] = quadrupoles[idx][3]
    quadrupole[1,2] = quadrupoles[idx][4]
    quadrupole[2,1] = quadrupoles[idx][4]
    quadrupole[2,2] = quadrupoles[idx][5]
    quadrupole = rotation_matrices[idx] @ quadrupole @ rotation_matrices[idx].T
    theta = [quadrupole[0,0], quadrupole[0,1], quadrupole[0,2], quadrupole[1,1], quadrupole[1,2], quadrupole[2,2]]
    print('{:<12d} {:16.12f} {:16.12f} {:16.12f} {:16.12f} {:16.12f} {:16.12f}'.format(atom.number, *theta))

print('@POLARIZABILITIES')
print('ORDER 1 1')
print(len(atoms))
for atom, p in zip(atoms, system.potential.values()):
    # for now, use standard cp3 polarizability
    print('{:<12d} {:16.12f} {:16.12f} {:16.12f} {:16.12f} {:16.12f} {:16.12f}'.format(atom.number, *p.P11))


print('EXCLISTS')
maxlength = 0
for p in system.potential.values():
    maxlength = max(maxlength, len(p.exclusion_list))

print(len(atoms), maxlength + 1)
fmt = '{:<8} ' +  ' {:>8d}'  * maxlength
for atom, p in zip(atoms, system.potential.values()):
    excluded = p.exclusion_list
    print(fmt.format(atom.number, *excluded, *[0]*(maxlength - len(excluded))))
