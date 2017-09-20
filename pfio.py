import dill
import os
import horton
import glob
import numpy as np
import sh

def save_file(thing, filename):
    with open(filename, "wb") as f:
        #we cannot save the obasis stuff
        try:
            del thing.obasis
        except:
            pass
        s = dill.dumps(thing)
        f.write(s)


def load_file(filename):
    with open(filename, "rb") as f:
        s = f.read()
    return dill.loads(s)

def loadfchks(regex):
    from chargefit import structure #sorry!
    fchks  = glob.glob(regex)
    structures = []
    for i in fchks:
        io = horton.IOData.from_file(i)
        try:
            grepfield = sh.grep("E-field", i, "-A1")
            field = np.array([float(x) for x in grepfield.stdout.split()[6:9]])
        except:
            print("INFO: no field information found in {}. Assuming zero field.".format(i))
            field = np.array([0., 0., 0.])
        structures.append(structure(io, fchkname=i, field=field))
    return structures
