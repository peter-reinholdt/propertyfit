#!/usr/bin/env python

import numpy as np
import horton
from pfio import save_file, load_file, load_qmfiles
from conversions import number2name, angstrom2bohr, bohr2angstrom
from numba import jit, float64


class structure(object):
    """
    A structure depends on horton for loading QM file data
    and calculating ESP on a grid.
    We define the grid-points on which to calculate ESP, as
    well as pre-calculated arrays of distances
    """
    def __init__(self, IO, fchkname='', field=np.zeros(3)):
        self.coordinates    = IO.coordinates
        self.numbers        = IO.numbers
        self.dm             = IO.get_dm_full()
        self.obasis         = IO.obasis
        self.natoms         = len(self.numbers)
        self.fchkname       = fchkname
        self.field          = field 


    def compute_grid_surface(self, pointdensity=2.0, radius_scale=1.4):
        """
        This part generates apparent uniformly spaced points on a vdW
        surface of a molecule.
        
        vdW     = van der Waals radius of atoms
        points  = number of points on a sphere around each atom
        grid    = output points in x, y, z
        idx     = used to keep track of index in grid, when generating 
                  initial points
        density = points per area on a surface
        chkrm   = (checkremove) used to keep track in index when 
                 removing points
        """
        vdW = {1:1.200*angstrom2bohr, 6:1.700*angstrom2bohr, 7:1.550*angstrom2bohr, 8:1.520*angstrom2bohr, 16:1.800*angstrom2bohr}
        points = np.zeros(len(self.numbers)-1)
        for i in range(1, len(self.numbers)):
            points[i-1] = int(pointdensity*4*np.pi*radius_scale*vdW[self.numbers[i]])
        # grid = [x, y, z]
        grid = np.zeros((np.int(np.sum(points)), 3))
        idx = 0
        for i in range(1, len(self.numbers)):
            N = int(points[i-1])
            #Saff & Kuijlaars algorithm
            for k in range(1, N+1):
                h = -1.0 +2.0*(k-1.0)/(N-1.0)
                theta = np.arccos(h)
                if k == 1 or k == N:
                    phi = 0
                else:
                    phi = ((phiold + 3.6/((N*(1-h**2))**0.5))) % (2*np.pi)
                phiold = phi
                x = radius_scale*vdW[self.numbers[i]]*np.cos(phi)*np.sin(theta)
                y = radius_scale*vdW[self.numbers[i]]*np.sin(phi)*np.sin(theta)
                z = radius_scale*vdW[self.numbers[i]]*np.cos(theta)
                grid[idx, 0] = x + self.coordinates[i,0]
                grid[idx, 1] = y + self.coordinates[i,1]
                grid[idx, 2] = z + self.coordinates[i,2]
                idx += 1
                
        # This is the distance points have to be apart
        dist = ((grid[0,0]-grid[1,0])**2+(grid[0,1]-grid[1,1])**2+(grid[0,2]-grid[1,2])**2)**0.5
        
        # Remove overlap all points to close to any atom
        for i in range(1, len(self.numbers)):
            chkrm = 0
            for j in range(0, len(grid)):
                r = ((grid[j-chkrm,0]-self.coordinates[i,0])**2+(grid[j-chkrm,1]-self.coordinates[i,1])**2+(grid[j-chkrm,2]-self.coordinates[i,2])**2)**0.5
                if r < radius_scale*0.99*vdW[self.numbers[i]]:
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
        return grid

    
    def compute_grid(self, rmin=1.4*angstrom2bohr, rmax=2.0*angstrom2bohr, pointdensity=1.0, nsurfaces=2):
        radii = np.linspace(rmin, rmax, nsurfaces)
        surfaces = []
        for r in radii:
            print(r)
            surfaces.append(self.compute_grid_surface(pointdensity=pointdensity, radius_scale=r))
        for s in surfaces:
            print(len(s))
        self.grid        = np.concatenate(surfaces)
        self.ngridpoints = len(self.grid)


    def compute_rinvmat(self):
        rinvmat = np.zeros((self.natoms, self.ngridpoints))
        for i in range(self.natoms):
            ri = self.coordinates[i]
            for j in range(self.ngridpoints):
                rj = self.grid[j]
                rinvmat[i,j] = np.sum((ri-rj)**2)**(-0.5)
        self.rinvmat = rinvmat


    def compute_xyzmat(self):
        xyzmat = np.zeros((self.natoms, self.ngridpoints, 3))
        for i in range(self.natoms):
            ri = self.coordinates[i]
            for j in range(self.ngridpoints):
                rj = self.grid[j]
                for k in range(3):
                    xyzmat[i,j,k] = ri[k] - rj[k]
        self.xyzmat = xyzmat


    def compute_qm_esp(self):
        esp_grid_qm = self.obasis.compute_grid_esp_dm(self.dm, self.coordinates, self.numbers.astype(float), self.grid)
        self.esp_grid_qm = esp_grid_qm 

   
    def compute_all(self):
        self.compute_grid()
        self.compute_rinvmat()
        self.compute_qm_esp()


    def write_xyz(self, filename):
        with open(filename, "w") as f:
            f.write("{}\n\n".format(self.natoms))
            for i in range(self.natoms):
                atomname = number2name[self.numbers[i]]
                f.write("{} {: .10f}   {: .10f}   {: .10f}\n".format(atomname, self.coordinates[i,0]*bohr2angstrom, self.coordinates[i,1]*bohr2angstrom,self.coordinates[i,2]*bohr2angstrom))

    def write_grid(self, filename):
        with open(filename, "w") as f:
            f.write("{}\n\n".format(self.ngridpoints))
            for i in range(self.ngridpoints):
                atomname = 'H'
                f.write("{} {: .10f}   {: .10f}   {: .10f}\n".format(atomname, self.grid[i,0]*bohr2angstrom, self.grid[i,1]*bohr2angstrom,self.grid[i,2]*bohr2angstrom))


    def reload_obasis(self):
        IO = horton.IOData.from_file(self.fchkname)
        self.obasis = IO.obasis


    def save(self, filename):
        save_file(self, filename)
