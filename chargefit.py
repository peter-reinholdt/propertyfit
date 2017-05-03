import numpy as np
import horton
import os
import sys


class structure(object):
    def __init__(self, IO):
        self.coordinates    = IO.coordinates
        self.numbers        = IO.numbers
        self.dm             = IO.get_dm_full()
        self.obasis         = IO.obasis


    def compute_grid(self, pointdensity=0.1):
        #TODO: Get Eriks code!
        self.grid = np.zeros((npoints,3))


    def compute_radii(self):
        pass


    def compute_qm_potential(self):
        pass


    def compute_ESP_squared_error(self, testcharges):
        pass



def loadfchks(dirname):
    content = os.listdir(dirname)
    fchks   = [f for f in content if ".fchk" in f]
    structures = []
    for i in fchks:
        io = horton.IOData.from_file(dirname + '/' + i)
        structures.append(structure(io))
        del io
    return structures

def cost():
    pass
    #her er det parallel
