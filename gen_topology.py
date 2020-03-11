#!/usr/bin/env python

import re
import json
import argparse
import numpy as np


def parse_range(index_list):
    indices = []
    string = "".join(index_list)
    terms = string.split(",")
    for term in terms:
        if "-" in term:
            start, end = [int(x) for x in term.split("-")]
            indices += list(range(start, end + 1))
        else:
            indices.append(int(term))
    return indices


parser = argparse.ArgumentParser()
parser.add_argument('--xyz', dest='xyz', type=str, required=True, help='Name of xyz file to read atom names from')
parser.add_argument('--fragment',
                    dest='frags',
                    type=str,
                    nargs='+',
                    action='append',
                    help='Specify atom indices for a fragment. Use multiple times to specify different\
        fragments. Example: --fragment 1,5,6-10')
parser.add_argument('--start-guess-charge',
                    '--charge',
                    dest='charge',
                    type=str,
                    help='Name of file with reference charges, used as start-guess and for\
        restraints',
                    required=True)
parser.add_argument('--start-guess-polarizability',
                    '--polarizability',
                    dest='polarizability',
                    type=str,
                    help='Name of file with reference polarizabilities, used as start-guess and for restraints')
parser.add_argument('--symmetry',
                    dest='syms',
                    type=str,
                    default=[],
                    nargs='+',
                    action='append',
                    help='Specify symmetry-equivalent charges')
parser.add_argument('--read-symmetry', dest='readsym', type=str, help='Read symmetries from newline delimited file')
parser.add_argument('--force-integer',
                    dest='force_integer',
                    type=bool,
                    default=True,
                    help='Turn off rounding and balacing of start-guess fragment charges')

args = parser.parse_args()

#read in input data
with open(args.xyz, "r") as f:
    lines = f.readlines()[2:]
atomnames = []
for line in lines:
    atomnames.append(line.split()[0])
with open(args.charge, "r") as f:
    start_guess_charge = [float(x) for x in f.readlines()]
mol_charge = 0.0
for q in start_guess_charge:
    mol_charge += q
for i in range(len(start_guess_charge)):
    start_guess_charge[i] = start_guess_charge[i] + (round(mol_charge) - mol_charge) / float(len(start_guess_charge))
if args.polarizability:
    with open(args.polarizability, "r") as f:
        start_guess_polarizability = [float(x) for x in f.readlines()]

syms = []
if args.readsym:
    lines = open(args.readsym, "r").readlines()
    for line in lines:
        syms.append([int(x) for x in line.split()])

out = dict()
out["name"] = args.xyz
out["fragments"] = []
for f in args.frags:
    frag_indices = parse_range(f)
    frag = dict()

    #get symmetry stuff
    frag["symmetries"] = []
    for symmetry in args.syms:
        sym_indices = parse_range(symmetry)
        if set(sym_indices).issubset(set(frag_indices)):
            frag["symmetries"].append(sym_indices)
    for symmetry in syms:
        if set(symmetry).issubset(set(frag_indices)):
            frag["symmetries"].append(symmetry)
    frag_atomnames = []
    frag_start_guess_charge = []
    frag_start_guess_polarizability = []
    qtot = 0.0
    for index in frag_indices:
        frag_atomnames.append(atomnames[index - 1])
        frag_start_guess_charge.append(start_guess_charge[index - 1])
        qtot += start_guess_charge[index - 1]
    n_atoms = len(frag_start_guess_charge)
    print(n_atoms, qtot)
    if args.force_integer:
        for i in range(n_atoms):
            frag_start_guess_charge[i] = frag_start_guess_charge[i] + (round(qtot) - qtot) / float(n_atoms)
        qtot = 0.0
        for i in range(n_atoms):
            qtot += frag_start_guess_charge[i]
        qtot = round(qtot)
    if args.polarizability:
        for index in frag_indices:
            frag_start_guess_polarizability.append(start_guess_polarizability[index - 1])

    frag["atomnames"] = frag_atomnames
    frag["atomindices"] = frag_indices
    frag["startguess_charge"] = frag_start_guess_charge
    frag["startguess_polarizability"] = frag_start_guess_polarizability
    frag["qtot"] = qtot
    out["fragments"].append(frag)

with open(args.xyz + ".constraints", "w") as f:
    json.dump(out, f, indent=4)
