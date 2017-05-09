import dill
import os
import horton
import glob
import numpy as np


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
        io = horton.IOData.from_file( i)
        structures.append(structure(io, i))
        del io
    return structures


def loadfchks_field(regex):
    from chargefit import structure
    fchks = glob.glob(regex) 
    structures = []
    for i in fchks:
        io = horton.IOData.from_file(i)
        field_string = i.split('/')[-1].split('.')[0].split('_')[-1]
        print(field_string)
        direction    = field_string[0]
        strength     = float(field_string[1:])
        if direction == 'x':
            field = np.array([strength, 0.0, 0.0])
        if direction == 'y':
            field = np.array([0.0, strength, 0.0])
        if direction == 'z':
            field = np.array([0.0, 0.0, strength])
        structures.append(structure(io, i, field))
        del io
    return structures


