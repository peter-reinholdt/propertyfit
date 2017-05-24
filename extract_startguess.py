#!/usr/bin/env python
import sys
import numpy as np


filename = sys.argv[1]
basename = filename.split("_")[0]
outfile  = "{}.x0".format(basename)

pols = []
with open(filename, "r") as f:
    while True:
        line = f.readline()
        if "@POLARIZABILITIES" in line:
            break
    f.readline()
    f.readline()
    while True:
        line = f.readline()
        if "EXCLISTS" in line:
            break
        else:
            pols.append(line.split())

pols = np.array(pols).astype(float)
isopol = (pols[:,1] + pols[:,4] + pols[:,6] ) / 3.0
np.savetxt(outfile, isopol)
