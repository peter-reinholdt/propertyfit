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
    
checkcap = 'None'
AAlist = ['ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'ALA', 'ASH', 'CYD', 'GLH', 'GLY', 'HID', 'HIE', 'LYD', 'LYS']
terminals = ['methylcharged','methylneutral','chargedmethyl','neutralmethyl', 'None']
#terminals = ['None']
#checkcap = 'NME'
for i in range(0, len(AAlist)):
    for j in range(0, len(terminals)):
        if terminals[j] == 'None':
            filename = AAlist[i]+'idx.csv'
        else:
            filename = AAlist[i]+terminals[j]+'idx.csv'
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
        elif "neutralmethyl" in name:
            nfrag = 2
            nf0 = 6
            nf2 = 0
        elif checkcap == 'NME':
            nfrag = 2
            nf0 = 6
            nf2 = 0
        elif checkcap == 'ACE':
            nfrag = 2
            nf0 = 0
            nf2 = 6
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
            if "chargedmethyl" in name:
                q0 = -1.0
            else:
                q0 = 0.0
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
            if 'LYS' in name or 'ARG' in name or 'HIS' in name:
                q0 = 1.0
            elif 'ASP' in name or 'GLU' in name or 'CYD' in name:
                q0 = -1.0
            else:
                q0 = 0.0
        
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
            if "methylcharged" in name:
                q0 = 1.0
            else:
                q0 = 0.0
        
            fragment = {"atomindices"   : list(atomindices),
                        "atomnames"     : list(atomnames),
                        "qtot"          : q0,
                        "symmetries"    : symmetries,
                        "startguess"    : [float(x) for x in qguess]}
            outdict["fragments"].append(fragment)
        
        stringout = json.dumps(outdict, indent=4, separators=(',', ': '))
        
        if checkcap == 'ACE':
            with open("{}ACE.constraints".format(name), "w") as f:
                f.write(stringout)
        elif checkcap == 'NME':
            with open("{}NME.constraints".format(name), "w") as f:
                f.write(stringout)
        else:
            with open("{}.constraints".format(name), "w") as f:
                f.write(stringout)
