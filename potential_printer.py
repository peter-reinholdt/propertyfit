#!/usr/bin/env python

# standalone script to assign CP3 anisotropic parameters to a pdb

import sys
import numpy as np
import pandas
import pyframe
from qcelemental import covalentradii
from propertyfit import rotations

parameters = pandas.read_csv("results.csv")
polarizability_parameters = pandas.read_csv("alpha_results.csv")
system = pyframe.MolecularSystem(sys.argv[1])
system.add_region('all',
                  fragments=system.fragments,
                  use_standard_potentials=True,
                  standard_potential_model='cp3',
                  standard_potential_exclusion_type='fragment')
project = pyframe.Project()
project.scratch_dir = '/tmp'
project.create_embedding_potential(system)
atoms = []
for fragment in system.fragments.values():
    atoms += fragment.atoms

charges = []
dipoles = []
quadrupoles = []
polarizabilities = []
rotation_matrices = []

equivalent_names = {
    'H': ['H1'],
    'H1': ['H'],
    'HA1': ['HA2', 'HA3'],  #HA1, HA2, HA3 equivalent
    'HA2': ['HA1', 'HA2'],
    'HA3': ['HA1', 'HA2'],
    'HB1': ['HB2', 'HB3'],  #HB1, HB2, HB3 equivalent
    'HB2': ['HB1', 'HB2'],
    'HB3': ['HB1', 'HB2'],
    'HG1': ['HG2', 'HG3'],  #HG1, HG2, HG3 equivalent
    'HG2': ['HG1', 'HG2'],
    'HG3': ['HG1', 'HG2'],
    'HD1': ['HD2', 'HD3'],  #HD1, HD2, HD3 equivalent
    'HD2': ['HD1', 'HD2'],
    'HD3': ['HD1', 'HD2'],
    'HE1': ['HE2', 'HE3'],  #HE1, HE2, HE3 equivalent
    'HE2': ['HE1', 'HE2'],
    'HE3': ['HE1', 'HE2'],
    'HG11': ['HG12', 'HG13'],  #HG11, HG12, HG13 equivalent
    'HG12': ['HG11', 'HG12'],
    'HG13': ['HG11', 'HG12'],
    'HG21': ['HG22', 'HG23'],  #HG21, HG22, HG23 equivalent
    'HG22': ['HG21', 'HG22'],
    'HG23': ['HG21', 'HG22'],
    'SD': ['SG'],  # SD/G in CYX
    'SG': ['SD'],
}

elements = []
coords = []
atomnames = []
for fragment in system.fragments.values():
    for atom in fragment.atoms:
        elements.append(atom.element)
        coords.append(atom.coordinate)
        atomnames.append(atom.name)
coords = np.array(coords)
bonds = []
for i in range(coords.shape[0]):
    bonded_atoms = []
    distances = np.linalg.norm(coords[i, :] - coords[:, :], axis=1)
    for idist, dist in enumerate(distances):
        if idist == i:
            continue
        if dist < 0.60 * (covalentradii.get(elements[i]) + covalentradii.get(elements[idist])):
            bonded_atoms.append(idist)
    bonds.append(bonded_atoms)

offset = 0
for fragment in system.fragments.values():
    residue = parameters[parameters.resname == fragment.name]
    aresidue = polarizability_parameters[polarizability_parameters.resname == fragment.name]
    fragment_atomnames = np.array([atom.name for atom in fragment.atoms])
    fragment_coordinates = np.array([atom.coordinate for atom in fragment.atoms])

    for iatom, atom in enumerate(fragment.atoms):
        if atom.name in residue.atomname.values:
            p = residue[residue.atomname == atom.name]
            ap = aresidue[aresidue.atomname == atom.name]
        else:
            for name in equivalent_names[atom.name]:
                if name in residue.atomname.values:
                    p = residue[residue.atomname == name]
                    ap = aresidue[aresidue.atomname == name]
                    break

        charges.append(p.charge.values[0])
        dipoles.append([p["dipole[0]"].values[0], p["dipole[1]"].values[0], p["dipole[2]"].values[0]])
        quadrupoles.append([
            p["quadrupole[0]"].values[0], p["quadrupole[1]"].values[0], p["quadrupole[2]"].values[0],
            p["quadrupole[3]"].values[0], p["quadrupole[4]"].values[0], p["quadrupole[5]"].values[0]
        ])
        polarizabilities.append([
            ap["polarizability[0]"].values[0], ap["polarizability[1]"].values[0], ap["polarizability[2]"].values[0],
            ap["polarizability[3]"].values[0], ap["polarizability[4]"].values[0], ap["polarizability[5]"].values[0]
        ])

        axis_atomnames = p["axis_atomnames"].values[0].split()
        if "internal" in p["axis_type"].values[0]:
            adjacent_indices_unordered = [b for b in bonds[atom.number - 1]]
            axis_indices = [atom.number - 1]
            for atomname in axis_atomnames:
                found_match = False
                for index in adjacent_indices_unordered:
                    if atomnames[index] == atomname:
                        found_match = True
                        axis_indices.append(index)
                        adjacent_indices_unordered.remove(index)
                        break
                if not found_match:
                    for equivalent_name in equivalent_names[atomname]:
                        for index in adjacent_indices_unordered:
                            if atomnames[index] == equivalent_name:
                                found_match = True
                                axis_indices.append(index)
                                adjacent_indices_unordered.remove(index)
                                break
                if not found_match:
                    raise ValueError
        elif "terminal" in p["axis_type"].values[0]:
            bonded_to = bonds[atom.number - 1][0]
            bonded_to_atomname = atomnames[bonded_to]
            adjacent_indices_unordered = [b for b in bonds[bonded_to] if b != atom.number - 1]
            axis_indices = [atom.number - 1, bonded_to]
            for atomname in axis_atomnames[1:]:
                found_match = False
                for index in adjacent_indices_unordered:
                    if atomnames[index] == atomname:
                        found_match = True
                        axis_indices.append(index)
                        adjacent_indices_unordered.remove(index)
                        break
                if not found_match:
                    for equivalent_name in equivalent_names[atomname]:
                        for index in adjacent_indices_unordered:
                            if atomnames[index] == equivalent_name:
                                found_match = True
                                axis_indices.append(index)
                                adjacent_indices_unordered.remove(index)
                                break
                if not found_match:
                    raise ValueError(f'{atom.name}, {axis_atomnames}')
        elif p["axis_type"].values[0] in ("zthenx", "bisector"):
            point1 = atom.coordinate
            point1_idx = iatom
            point2_atomnames = [p["axis_atomnames"].values[0].split()[0]]
            extra_names = []
            for name in point2_atomnames:
                if name in equivalent_names:
                    extra_names += equivalent_names[name]
            all_names = np.array([list(set(point2_atomnames + extra_names))])
            locs = np.where(all_names.T == fragment_atomnames)
            point2_idx = list(set(locs[1][:]).difference(set([point1_idx])))[0]
            point2 = fragment_coordinates[point2_idx]
            point3_atomnames = [p["axis_atomnames"].values[0].split()[1]]
            extra_names = []
            for name in point3_atomnames:
                if name in equivalent_names:
                    extra_names += equivalent_names[name]
            all_names = np.array([list(set(point3_atomnames + extra_names))])
            locs = np.where(all_names.T == fragment_atomnames)
            point3_idx = list(set(locs[1][:]).difference(set([point1_idx, point2_idx])))[0]
            point3 = fragment_coordinates[point3_idx]
            axis_indices = [idx + offset for idx in (point1_idx, point2_idx, point3_idx)]
        else:
            raise ValueError

        points = [*[coords[i, :] for i in axis_indices]]
        R = getattr(rotations, p["axis_type"].values[0])(*points)
        rotation_matrices.append(R)
    offset += len(fragment_atomnames)

print('@COORDINATES')
print(len(atoms))
print('AA')
for atom in atoms:
    print('{:6s} {:16.12f} {:16.12f} {:16.12f} {:12d} ! {}'.format(atom.element, *atom.coordinate, atom.number,
                                                                   atom.name))

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
    quadrupole = np.zeros((3, 3))
    quadrupole[0, 0] = quadrupoles[idx][0]
    quadrupole[0, 1] = quadrupoles[idx][1]
    quadrupole[1, 0] = quadrupoles[idx][1]
    quadrupole[0, 2] = quadrupoles[idx][2]
    quadrupole[2, 0] = quadrupoles[idx][2]
    quadrupole[1, 1] = quadrupoles[idx][3]
    quadrupole[1, 2] = quadrupoles[idx][4]
    quadrupole[2, 1] = quadrupoles[idx][4]
    quadrupole[2, 2] = quadrupoles[idx][5]
    quadrupole = rotation_matrices[idx] @ quadrupole @ rotation_matrices[idx].T
    theta = [
        quadrupole[0, 0], quadrupole[0, 1], quadrupole[0, 2], quadrupole[1, 1], quadrupole[1, 2], quadrupole[2, 2]
    ]
    print('{:<12d} {:16.12f} {:16.12f} {:16.12f} {:16.12f} {:16.12f} {:16.12f}'.format(atom.number, *theta))

print('@POLARIZABILITIES')
print('ORDER 1 1')
print(len(atoms))
for idx, atom in enumerate(atoms):
    # standard cp3 polarizability
    # print('{:<12d} {:16.12f} {:16.12f} {:16.12f} {:16.12f} {:16.12f} {:16.12f}'.format(atom.number, *p.P11))
    # anisotropic cp3 -- unpack, rotate, repack
    polarizability = np.zeros((3, 3))
    polarizability[0, 0] = polarizabilities[idx][0]
    polarizability[0, 1] = polarizabilities[idx][1]
    polarizability[1, 0] = polarizabilities[idx][1]
    polarizability[0, 2] = polarizabilities[idx][2]
    polarizability[2, 0] = polarizabilities[idx][2]
    polarizability[1, 1] = polarizabilities[idx][3]
    polarizability[1, 2] = polarizabilities[idx][4]
    polarizability[2, 1] = polarizabilities[idx][4]
    polarizability[2, 2] = polarizabilities[idx][5]
    polarizability = rotation_matrices[idx] @ polarizability @ rotation_matrices[idx].T
    alpha = [
        polarizability[0, 0], polarizability[0, 1], polarizability[0, 2], polarizability[1, 1], polarizability[1, 2],
        polarizability[2, 2]
    ]
    print('{:<12d} {:16.12f} {:16.12f} {:16.12f} {:16.12f} {:16.12f} {:16.12f}'.format(atom.number, *alpha))

print('EXCLISTS')
maxlength = 0
for p in system.potential.values():
    maxlength = max(maxlength, len(p.exclusion_list))

print(len(atoms), maxlength + 1)
fmt = '{:<8} ' + ' {:>8d}' * maxlength
for atom, p in zip(atoms, system.potential.values()):
    excluded = p.exclusion_list
    print(fmt.format(atom.number, *excluded, *[0] * (maxlength - len(excluded))))
