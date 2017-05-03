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


    def compute_grid(self, pointdensity=1.0):
        # #####################################################
        #
        # This part generates apparent uniformly spaced points on a vdW
        # surface of a molecule.
        #
        # vdW     = van der Waals radius of atoms
        # points  = number of points on a sphere around each atom
        # grid    = output points in x, y, z
        # idx     = used to keep track of index in grid, when generating 
        #           initial points
        # density = points per area on a surface
        # chkrm   = (checkremove) used to keep track in index when 
        #           removing points
        #
        # #####################################################
        # vdW in pm as of now
        vdW = {1:120.0, 6:170.0, 7:155.0, 8:152.0, 16:180.0}
        points = np.zeros(len(self.numbers)-1)
        for i in range(1, len(self.numbers)):
            points[i-1] = int(pointdensity*4*np.pi*1.4*vdW[self.numbers[i]])
        # grid = [x, y, z]
        grid = np.zeros((np.int(np.sum(points)), 3))
        idx = 0
        for i in range(1, len(self.numbers)):
            N = int(points[i-1])
            #Saff & Kuijlaars algorithm
            for k in range(1, N+1):
                h = -1 +2*(k-1)/(N-1)
                theta = np.arccos(h)
                if k == 1 or k == N:
                    phi = 0
                else:
                    phi = ((phiold + 3.6/((N*(1-h**2))**0.5))) % (2*np.pi)
                phiold = phi
                x = 1.4*vdW[self.numbers[i]]*np.cos(phi)*np.sin(theta)
                y = 1.4*vdW[self.numbers[i]]*np.sin(phi)*np.sin(theta)
                z = 1.4*vdW[self.numbers[i]]*np.cos(theta)
                grid[idx, 0] = x + self.coordinates[i,0]
                grid[idx, 1] = y + self.coordinates[i,1]
                grid[idx, 2] = z + self.coordinates[i,2]
                idx += 1
                
        # This is the distance points have to be apart
        dist = ((grid[0,0]-grid[1,0])**2+(grid[0,1]-grid[1,1])**2+(grid[0,2]-grid[1,2])**2)**0.5
        
        # Remove overlap all points to close to any atom
        for i in range(1, len(input)):
            chkrm = 0
            for j in range(0, len(grid)):
                r = ((grid[j-chkrm,0]-self.coordinates[i,0])**2+(grid[j-chkrm,1]-self.coordinates[i,1])**2+(grid[j-chkrm,2]-self.coordinates[i,2])**2)**0.5
                if r < 1.39*vdW[self.numbers[i]]:
                    grid = np.delete(grid,j-chkrm,axis=0)
                    chkrm += 1
        chkrm = 0
        # Double loop over grid to remove close lying points
        for i in range(0, len(grid)):
            for j in range(0, len(grid)):
                if 0.9*dist > ((grid[i-chkrm,0]-grid[j,0])**2+(grid[i-chkrm,1]-grid[j,1])**2+(grid[i-chkrm,2]-grid[j,2])**2)**0.5 and i-chkrm != j:
                    grid = np.delete(grid,j,axis=0)
                    chkrm += 1
                    break
        
        self.grid


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
