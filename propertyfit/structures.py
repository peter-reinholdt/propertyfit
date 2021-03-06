#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division, print_function
import numpy as np
import warnings
try:
    import horton
except:
    warnings.warn("Running without support for horton", RuntimeWarning)
import h5py
from .utilities import load_qmfiles, number2name, name2number, angstrom2bohr, bohr2angstrom, load_json, vdw_radii, load_geometry_from_molden
from numba import jit
import os
import sh



class structure(object):
    """
    A structure depends on horton for loading QM file data
    and calculating ESP on a grid.
    We define the grid-points on which to calculate ESP, as
    well as pre-calculated arrays of distances
    """
    def __init__(self, vdw_grid_rmin=1.4, vdw_grid_rmax=2.0, vdw_grid_pointdensity=2.0, vdw_grid_nsurfaces=2):
        self.vdw_grid_rmin = vdw_grid_rmin
        self.vdw_grid_rmax = vdw_grid_rmax
        self.vdw_grid_pointdensity = vdw_grid_pointdensity
        self.vdw_grid_nsurfaces = vdw_grid_nsurfaces
        self.dm = None


    def load_qm(self, filename, field=np.zeros(3, dtype=np.float64)):
        IO                  = horton.IOData.from_file(filename)
        self.coordinates    = IO.coordinates
        self.numbers        = IO.numbers
        self.dm             = IO.get_dm_full()
        self.obasis         = IO.obasis
        self.natoms         = len(self.numbers)
        self.fchkname       = filename
        self.field          = field

    def load_esp_terachem(self, terachem_scrdir, field=np.zeros(3, dtype=np.float64)):
        #we really only need to provide coordinates, grid_points, (external field), and ESP
        #from terachem manual:
        #scr/esp.xyz – The ESP grid points in Å, together with the ESP on that point. Each row stands for
        #   one grid point.
        #   Colunm 1: The element type of the atom that the grid point originates from.
        #   Column 2-4: coordinates of the grid point
        #   Column 5: Electrostatic potential (ESP) on that grid point.
        #   Column 6: The index of the the atom that the grid point originates from. Order of the index is
        #   the same as the molecule in the input deck.
        #   When we use software to visualize this xyz file, only data in the first 4 columns is read by the software,
        #   though sometimes the 5th column can also be recognized and presents in labels (Molden).
        esp_data = np.loadtxt(terachem_scrdir + '/esp.xyz', skiprows=2, dtype=str)[:,1:5].astype(np.float64)
        self.grid = esp_data[:,0:3] * angstrom2bohr
        self.esp_grid_qm = esp_data[:,3] #quite sure this is in hartree
        self.ngridpoints = self.esp_grid_qm.shape[0]
        #we assume xyz is in angstrom and convert to bohr
        molden_filename = terachem_scrdir + '/' +  [name for name in  os.listdir(terachem_scrdir) if '.molden' in name][0]
        self.coordinates, elements = load_geometry_from_molden(molden_filename)
        self.numbers = np.array([name2number[el] for el in elements], dtype=np.int64)
        self.natoms = self.coordinates.shape[0]

    def load_esp_orca(self, gbw_file, density_file, field=np.zeros(3, dtype=np.float64)):
        #we generate our own grid and run 
        # orca_vpot  GBWName PName XYZName POTName
        #  GBWName  = GBW file that contains the orbitals/coordinates/basis
        #  PName    = File that contains the density (must match the GBW file basis set!); for HF/DFT ground state jobname.scfp; for tddft jobname.cisp etc...
        #  XYZName  = File that contains the coordinates to evaluate V(r) for
        #  POTName  = Output file with V(r)
        
        self.field = field
        #0) read in atomic positions and elements
        #convert gbw to molden
        gbw_base = ".".join(gbw_file.split(".")[:-1])
        sh.orca_2mkl(gbw_base)
        sh.orca_2mkl(gbw_base, "-molden")
        self.coordinates, elements = load_geometry_from_molden(gbw_base  + ".molden.input")
        self.numbers = np.array([name2number[el] for el in elements], dtype=np.int64)
        self.natoms = len(self.numbers)


        #1) Generate grid and write it out to file
        grid_file = gbw_base + ".grid"
        esp_file  = gbw_base + ".esp"
        self.compute_grid(rmin=self.vdw_grid_rmin, rmax=self.vdw_grid_rmax, pointdensity=self.vdw_grid_pointdensity, nsurfaces=self.vdw_grid_nsurfaces)
        np.savetxt(grid_file, self.grid, header=str(self.grid.shape[0]), comments=" ")

        #2) Run orca_vpot to get esp on grid
        
        sh.orca_vpot(gbw_file, density_file, grid_file, esp_file)

        #3) Read in esp
        #rx, ry, rz, esp(r)
        esp = np.loadtxt(esp_file, skiprows=1)[:,3]
        self.esp_grid_qm = esp
        self.compute_rinvmat()
        self.compute_xyzmat()



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
        print(rmin, rmax, pointdensity, nsurfaces)
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
        self.rinvmat = 1./np.sqrt(np.sum((self.coordinates[:,np.newaxis,:] - self.grid[np.newaxis,:,:])**2, axis=2))


    def compute_xyzmat(self):
        self.xyzmat = self.coordinates[:,np.newaxis,:] - self.grid[np.newaxis,:,:]


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
        f.create_dataset("coordinates", data=self.coordinates)
        f.create_dataset("numbers",     data=self.numbers)
        f.create_dataset("natoms",      data=self.natoms)
        f.create_dataset("field",       data=self.field)
        f.create_dataset("xyzmat",      data=self.xyzmat)
        f.create_dataset("rinvmat",     data=self.rinvmat)
        f.create_dataset("esp_grid_qm", data=self.esp_grid_qm)
        f.close()


    def load_h5(self, filename):
        f = h5py.File(filename, "r")
        self.coordinates    = f["coordinates"][()]
        self.numbers        = f["numbers"][()]
        self.natoms         = f["natoms"][()]
        self.field          = f["field"][()]
        self.xyzmat         = f["xyzmat"][()]
        self.rinvmat        = f["rinvmat"][()]
        self.esp_grid_qm    = f["esp_grid_qm"][()]
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
