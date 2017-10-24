#!/usr/bin/env python

import sys
import numpy as np
import json

def getSymmetries(lista, listb):
    symmetries = dict()

    for i in range(len(lista)):
        if listb[i] != lista[i]:
            try:
                symmetries[listb[i]].append(lista[i])
            except KeyError:
                symmetries[listb[i]] = [listb[i], lista[i]]
    return list(symmetries.values())
    

filename    = sys.argv[1]
name        = filename.split("idx")[0]
data        = np.loadtxt(filename, dtype="U64", delimiter=";")
natoms = len(data)

if "methylneutral" in name:
    nfrag = 2
    nf0 = 0
    nf2 = 6
elif "methylcharged" in name:
    nfrag = 2
    nf0 = 0
    nf2 = 6
elif "chargedmethyl" in name:
    nfrag = 2
    nf0 = 6
    nf2 = 0
elif "chargedneutral" in name:
    nfrag = 2
    nf0 = 6
    nf2 = 0
else:
    nfrag = 3
    nf0   = 6
    nf2   = 6

nf1 = len(data) - nf0 - nf2

outdict = {"name": name, "fragments": []}


if nf0 > 0:
    start = 0
    stop = nf0
    atomnames   = data[start:stop, 0]
    atomindices = [int(x) for x in data[start:stop, 1]]
    symidx      = [int(x) for x in data[start:stop, 2]]
    symmetries  = getSymmetries(atomindices, symidx)
    qguess      = data[start:stop, 3].astype(np.float64)
    q0          = np.round(np.sum(qguess))
    #get symmetry
    

    fragment = {"atomindices"   : list(atomindices),
                "atomnames"     : list(atomnames),
                "qtot"          : q0,
                "symmetries"    : symmetries,
                "startguess"    : [float(x) for x in qguess]}
    outdict["fragments"].append(fragment)

if nf1 > 0:
    start = nf0
    stop = nf0 + nf1
    atomnames   = data[start:stop, 0]
    atomindices = [int(x) for x in data[start:stop, 1]]
    symidx      = [int(x) for x in data[start:stop, 2]]
    symmetries  = getSymmetries(atomindices, symidx)
    qguess      = data[start:stop, 3].astype(np.float64)
    q0          = np.round(np.sum(qguess))

    fragment = {"atomindices"   : list(atomindices),
                "atomnames"     : list(atomnames),
                "qtot"          : q0,
                "symmetries"    : symmetries,
                "startguess"    : [float(x) for x in qguess]}
    outdict["fragments"].append(fragment)

if nf2 > 0:
    start = nf0 + nf1
    stop = nf0 + nf1 + nf2
    atomnames   = data[start:stop, 0]
    atomindices = [int(x) for x in data[start:stop, 1]]
    symidx      = [int(x) for x in data[start:stop, 2]]
    symmetries  = getSymmetries(atomindices, symidx)
    qguess      = data[start:stop, 3].astype(np.float64)
    q0          = np.round(np.sum(qguess))

    fragment = {"atomindices"   : list(atomindices),
                "atomnames"     : list(atomnames),
                "qtot"          : q0,
                "symmetries"    : symmetries,
                "startguess"    : [float(x) for x in qguess]}
    outdict["fragments"].append(fragment)

stringout = json.dumps(outdict, indent=4, separators=(',', ': '))

with open("{}.constraints".format(name), "w") as f:
    f.write(stringout)
