#!/usr/bin/env python

import chargefit
import polfit
import scipy.optimize as opt
import numpy as np
import glob
import sys
import re

aafolder = sys.argv[1]
AA       = sys.argv[2] # Might be same as aafolder?
alphatot = sys.argv[3]

#find and match "unfielded" to "fielded" versions of a particular structure
frsx = glob.glob(aafolder+"/*.fchk.s")
frs = []
ffs = []
for i in frsx:
    x = re.split("/|\.", i)[-3]
    print(x)
    for j in glob.glob(aafolder+"/field/*"+x+"_*.fchk.s"):
        ffs.append(j)
        frs.append(i)

rs = [chargefit.load_file(f) for f in frs]
fs = [chargefit.load_file(f) for f in ffs]

a = np.random.rand(rs[0].natoms)

def fun(alpha):
    alpha_in = alpha
    # Make same atoms same polarizability
    for i in range(0, len(constraints)):
        alpha_in[i] = alpha_in[int(constraints[i,2])-1]
    
    # Total molecular polarizability set to alphatot
    alpha_in[:] -= (alpha_in[:].sum() - alphatot)/len(alpha_in)
    
    res =  polfit.cost_alpha_iso(rs, fs, alpha_in)*2625.5002
    print(res)
    return res

def return_alpha(alpha):
    # Not sure this function is needed
    alpha_in = alpha
    # Make same atoms same polarizability
    for i in range(0, len(constraints)):
        alpha_in[i] = alpha_in[int(constraints[i,2])-1]
    
    # Total molecular polarizability set to alphatot
    if alphatot != 'ignore':
        alpha_in[:] -= (alpha_in[:].sum() - alphatot)/len(alpha_in)
    return alpha_in

# Ugly way to load in constraints, but it works for now
constraints = np.genfromtxt('constraints/'+AA+'idx.csv', delimiter=';', dtype=str)
for i in range(0, len(constraints)):
        constraints[i,1] = int(constraints[i,1])
        constraints[i,2] = int(constraints[i,2])
        constraints[i,3] = float(constraints[i,3])
        #constraints[i,4] = float(constraints[i,4]) # Contains initial guess for alpha

#Check if sys arg and fchk match
check = 0
for i in rs:
    for j in range(0, len(i.numbers)):
        if i.numbers[j] != name2number[constraints[j,0]]:
            check = 1
            print('FATAL ERROR: Atom number '+str(j+1)+' does not match')
            break

if check == 0:
    res = opt.minimize(fun, x0=a, method="bfgs")
    print(res)
    print(return_alpha(res['x'])
else:
    print('FATAL ERROR')
