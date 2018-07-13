#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import horton
import h5py
from utilities import load_qmfiles, number2name, angstrom2bohr, bohr2angstrom, load_json, vdw_radii
from numba import jit



class structure(object):
    """
    A structure depends on horton for loading QM file data
    and calculating ESP on a grid.
    We define the grid-points on which to calculate ESP, as
    well as pre-calculated arrays of distances
    """
    def __init__(self):
        pass


    def load_qm(self, filename, field):
        IO                  = horton.IOData.from_file(filename)
        self.coordinates    = IO.coordinates
        self.numbers        = IO.numbers
        self.dm             = IO.get_dm_full()
        self.obasis         = IO.obasis
        self.natoms         = len(self.numbers)
        self.fchkname       = filename
        self.field          = field 


    def compute_grid_surface(self, pointdensity=2.0, radius_scale=1.4):
        """
        Generates apparent uniformly spaced points on a vdw_radii
        surface of a molecule.
        
        vdw_radii   = van der Waals radius of atoms
        points      = number of points on a sphere around each atom
        grid        = output points in x, y, z
        idx         = used to keep track of index in grid, when generating 
                      initial points
        density     = points per area on a surface
        chkrm       = (checkremove) used to keep track in index when 
                      removing points
        """
        points = np.zeros(self.natoms, dtype=np.int64)
        for i in range(self.natoms):
            points[i] = np.int(pointdensity*4*np.pi*radius_scale*vdw_radii[self.numbers[i]])
        # grid = [x, y, z]
        grid = np.zeros((np.sum(points), 3), dtype=np.float64)
        idx = 0
        for i in range(self.natoms):
            N = points[i]
            #Saff & Kuijlaars algorithm
            for k in range(N):
                h = -1.0 +2.0*k/(N-1)
                theta = np.arccos(h)
                if k == 0 or k == (N-1):
                    phi = 0.0
                else:
                    #phi_k  phi_{k-1}
                    phi = ((phi + 3.6/np.sqrt(N*(1-h**2)))) % (2*np.pi)
                x = radius_scale*vdw_radii[self.numbers[i]]*np.cos(phi)*np.sin(theta)
                y = radius_scale*vdw_radii[self.numbers[i]]*np.sin(phi)*np.sin(theta)
                z = radius_scale*vdw_radii[self.numbers[i]]*np.cos(theta)
                grid[idx, 0] = x + self.coordinates[i,0]
                grid[idx, 1] = y + self.coordinates[i,1]
                grid[idx, 2] = z + self.coordinates[i,2]
                idx += 1
                
        dist = lambda i,j: np.sqrt(np.sum((i-j)**2))
        
        #This is the distance points have to be apart
        #since they are from the same atom
        grid_spacing = dist(grid[0,:], grid[1,:])
        
        #Remove overlap all points to close to any atom
        not_near_atom = np.ones(grid.shape[0], dtype=bool)
        for i in range(self.natoms):
            for j in range(grid.shape[0]):
                r = dist(grid[j,:], self.coordinates[i,:])
                if r < radius_scale*0.99*vdw_radii[self.numbers[i]]:
                    not_near_atom[j] = False
        grid = grid[not_near_atom]
    
        # Double loop over grid to remove close lying points
        not_overlapping = np.ones(grid.shape[0], dtype=bool)
        for i in range(grid.shape[0]):
            for j in range(i+1, grid.shape[0]):
                if (not not_overlapping[j]): continue #already marked for removal
                r = dist(grid[i,:], grid[j,:])
                if 0.90 * grid_spacing > r:
                    not_overlapping[j] = False
        grid = grid[not_overlapping]
        return grid

    
    def compute_grid(self, rmin=1.4, rmax=2.0, pointdensity=1.0, nsurfaces=2):
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


    def save_h5(self, filename):
        """
        Save important arrays
        on disk
        """
        f = h5py.File(filename, "w")
        dmgroup = f.create_group("dm")
        self.dm.to_hdf5(dmgroup)
        f.create_dataset("nbasis",      data=self.dm.nbasis)
        f.create_dataset("coordinates", data=self.coordinates)
        f.create_dataset("numbers",     data=self.numbers)
        f.create_dataset("natoms",      data=self.natoms)
        f.create_dataset("fchkname",    data=self.fchkname)
        f.create_dataset("field",       data=self.field)
        f.create_dataset("xyzmat",      data=self.xyzmat)
        f.create_dataset("rinvmat",     data=self.rinvmat)
        f.create_dataset("esp_grid_qm", data=self.esp_grid_qm)
        f.close()


    def load_h5(self, filename):
        f = h5py.File(filename, "r")
        self.dm = horton.matrix.dense.DenseTwoIndex(f["nbasis"].value)
        self.dm.from_hdf5(f["dm"])
        self.coordinates    = f["coordinates"].value
        self.numbers        = f["numbers"].value
        self.natoms         = f["natoms"].value
        self.fchkname       = f["fchkname"].value
        self.field          = f["field"].value
        self.xyzmat         = f["xyzmat"].value
        self.rinvmat        = f["rinvmat"].value
        self.esp_grid_qm    = f["esp_grid_qm"].value
        f.close()



class fragment(object):
    def __init__(self, fragdict):
        self.atomindices    = np.array(fragdict["atomindices"],dtype=np.int64) - 1
        self.atomnames      = fragdict["atomnames"]
        self.qtot           = fragdict["qtot"]
        self.symmetries     = [list(np.array(x, dtype=np.int64) - 1) for x in fragdict["symmetries"]]
        self.fullsymmetries = []
        self.natoms         = len(self.atomindices)
        self.symmetryidx    = np.copy(self.atomindices)
        self.nparamtersq    = 0
        self.nparamtersa    = 0
        self.lastidx        = self.atomindices[-1]
        self.lastidxissym   = False
        self.lastidxnsym    = 1  #standard, no symmetry on last atom 
        self.lastidxsym     = [self.lastidx]
        self.startguess_charge = fragdict["startguess_charge"]
        self.startguess_polarizability = fragdict["startguess_polarizability"]


        for iloc, idx in enumerate(self.symmetryidx):
            for sym in self.symmetries:
                if idx in sym:
                    self.symmetryidx[iloc] = sym[0]
                    if idx == self.lastidx:
                        self.lastidxissym  = True
                        self.lastidxsym    = sym
                        self.lastidxnsym   = len(sym)
        
        self.fullsymmetries = []
        for idx in self.atomindices:
            counted = False
            for sym in self.fullsymmetries:
                if idx in sym:
                    counted = True
            if counted:
                continue

            insym = False
            for sym in self.symmetries:
                if idx in sym:
                    insym = True
                    break
            if insym:
                self.fullsymmetries.append(sym)
            else:
                self.fullsymmetries.append([idx])
        

        #number of paramters less than the total amount
        # due to symmetries
        nsymp = 0
        for sym in self.symmetries:
            nsymp += len(sym) - 1
        #Np              = Na          - nsym  - (sum constraint)
        self.nparametersq = self.natoms - nsymp - 1
        #for isotropic polarizability, there is no constraint on
        # the sum
        self.nparametersa = self.natoms - nsymp
          


class constraints(object):
    def __init__(self, filename):
        data = load_json(filename)
        self.filename       = filename
        self.name           = data["name"]
        self.restraint      = 0.0
        self.nfragments     = len(data["fragments"])
        self.fragments      = []
        self.qtot           = 0.0
        self.natoms         = 0
        self.nparametersq   = 0
        self.nparametersa   = 0
        q_red   = []
        a_red   = []
        indices = []

        for i in range(self.nfragments):
            frag = fragment(data["fragments"][i])
            self.qtot           += frag.qtot
            self.natoms         += frag.natoms
            self.nparametersq   += frag.nparametersq
            self.nparametersa   += frag.nparametersa
            q_red               += frag.startguess_charge         #redundant start guesses
            a_red               += frag.startguess_polarizability #redundant start guesses
            self.fragments.append(frag)
       
        #get non-redundant start guess
        #1) remove (symmetry) indices from end
        indices = []
        for frag in self.fragments:
            for sym in frag.fullsymmetries[:-1]:
                indices.append(sym[0])
                q_sym = 0.0
                for member in sym:
                    q_sym += q_red[member]
                q_sym = q_sym / len(sym)
                for member in sym:
                    q_red[member] = q_sym
        
        q_red   = np.array(q_red,               dtype=np.float64)
        self.q0 = np.zeros(self.nparametersq,   dtype=np.float64)

        for i, index in enumerate(indices):
            self.q0[i] = q_red[index]
        #same, but for polarizability.
        #there is no constraint on the total polarizability, just do the symmetry part
        indices = []
        if a_red:
            for frag in self.fragments:
                for sym in frag.fullsymmetries[:]:
                    indices.append(sym[0])
                    a_sym = 0.0
                    for member in sym:
                        a_sym += a_red[member]
                    a_sym = a_sym / len(sym)
                    for member in sym:
                        a_red[member] = a_sym
            
            a_red   = np.array(a_red,               dtype=np.float64)
            self.a0 = np.zeros(self.nparametersa,   dtype=np.float64)
            for i, index in enumerate(indices):
                self.a0[i] = a_red[index]

    def expand_q(self, qcompressed):
        qout = np.zeros(self.natoms, dtype=np.float64)
        pcounter = 0
        for frag in self.fragments:
            qcur = 0.0
            for sym in frag.fullsymmetries[:-1]:
                for idx in sym:
                    qout[idx] = qcompressed[pcounter]
                    qcur += qout[idx]
                pcounter += 1
            #charge constraint. lastidxnsym is 1 if the last one is not a part of a symmetry
            qlast = (frag.qtot - qcur) / len(frag.fullsymmetries[-1])
            for idx in frag.fullsymmetries[-1]:
                qout[idx] = qlast 
        return qout

    def expand_a(self, acompressed):
        aout = np.zeros((self.natoms,3,3), dtype=np.float64)
        pcounter = 0
        for frag in self.fragments:
            for sym in frag.fullsymmetries[:]:
                for idx in sym:
                    aout[idx,0,0] = acompressed[pcounter]
                    aout[idx,1,1] = acompressed[pcounter]
                    aout[idx,2,2] = acompressed[pcounter]
                pcounter += 1
        return aout
