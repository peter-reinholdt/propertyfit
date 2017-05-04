import dill
import os
import horton


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

def loadfchks(dirname):
    from chargefit import structure #sorry!
    content = os.listdir(dirname)
    fchks   = [f for f in content if ".fchk" in f]
    structures = []
    for i in fchks:
        io = horton.IOData.from_file(dirname + '/' + i)
        structures.append(structure(io, dirname + '/' + i))
        del io
    return structures
