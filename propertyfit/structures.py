#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import warnings

import sh
import h5py
import numpy as np

from qcelemental import periodictable, vdwradii
from qcelemental import constants

from . import rotations
from .utilities import load_qmfiles, load_json, load_geometry_from_molden, dipole_axis_nonzero, quadrupole_axis_nonzero
from .potentials import T0, T1, T2

try:
    import horton
except:
    pass
    #warnings.warn("Running without support for horton", RuntimeWarning)


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
        self._T0 = None
        self._T1 = None
        self._T2 = None
        self.rotation_matrices = None

    def load_qm(self, filename, field=np.zeros(3, dtype=np.float64)):
        IO = horton.IOData.from_file(filename)
        self.coordinates = IO.coordinates
        self.numbers = IO.numbers
        self.dm = IO.get_dm_full()
        self.obasis = IO.obasis
        self.natoms = len(self.numbers)
        self.fchkname = filename
        self.field = field

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
        esp_data = np.loadtxt(terachem_scrdir + '/esp.xyz', skiprows=2, dtype=str)[:, 1:5].astype(np.float64)
        self.grid = esp_data[:, 0:3] / constants.bohr2angstrom
        self.esp_grid_qm = esp_data[:, 3]  #quite sure this is in hartree
        self.ngridpoints = self.esp_grid_qm.shape[0]
        #we assume xyz is in angstrom and convert to bohr
        molden_filename = terachem_scrdir + '/' + [name
                                                   for name in os.listdir(terachem_scrdir) if '.molden' in name][0]
        self.coordinates, elements = load_geometry_from_molden(molden_filename)
        self.numbers = np.array([periodictable.to_atomic_number[el] for el in elements], dtype=np.int64)
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
        self.coordinates, elements = load_geometry_from_molden(gbw_base + ".molden.input")
        self.numbers = np.array([periodictable.to_atomic_number[el] for el in elements], dtype=np.int64)
        self.natoms = len(self.numbers)

        #1) Generate grid and write it out to file
        grid_file = gbw_base + ".grid"
        esp_file = gbw_base + ".esp"
        self.compute_grid(rmin=self.vdw_grid_rmin,
                          rmax=self.vdw_grid_rmax,
                          pointdensity=self.vdw_grid_pointdensity,
                          nsurfaces=self.vdw_grid_nsurfaces)
        np.savetxt(grid_file, self.grid, header=str(self.grid.shape[0]), comments=" ")

        #2) Run orca_vpot to get esp on grid

        sh.orca_vpot(gbw_file, density_file, grid_file, esp_file)

        #3) Read in esp
        #rx, ry, rz, esp(r)
        esp = np.loadtxt(esp_file, skiprows=1)[:, 3]
        self.esp_grid_qm = esp

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
            points[i] = np.int(pointdensity * 4 * np.pi * radius_scale * vdw_radii.get(self.numbers[i]))
        # grid = [x, y, z]
        grid = np.zeros((np.sum(points), 3), dtype=np.float64)
        idx = 0
        for i in range(self.natoms):
            N = points[i]
            #Saff & Kuijlaars algorithm
            for k in range(N):
                h = -1.0 + 2.0 * k / (N - 1)
                theta = np.arccos(h)
                if k == 0 or k == (N - 1):
                    phi = 0.0
                else:
                    #phi_k  phi_{k-1}
                    phi = ((phi + 3.6 / np.sqrt(N * (1 - h**2)))) % (2 * np.pi)
                x = radius_scale * vdw_radii.get(self.numbers[i]) * np.cos(phi) * np.sin(theta)
                y = radius_scale * vdw_radii.get(self.numbers[i]) * np.sin(phi) * np.sin(theta)
                z = radius_scale * vdw_radii.get(self.numbers[i]) * np.cos(theta)
                grid[idx, 0] = x + self.coordinates[i, 0]
                grid[idx, 1] = y + self.coordinates[i, 1]
                grid[idx, 2] = z + self.coordinates[i, 2]
                idx += 1

        dist = lambda i, j: np.sqrt(np.sum((i - j)**2))

        #This is the distance points have to be apart
        #since they are from the same atom
        grid_spacing = dist(grid[0, :], grid[1, :])

        #Remove overlap all points to close to any atom
        not_near_atom = np.ones(grid.shape[0], dtype=bool)
        for i in range(self.natoms):
            for j in range(grid.shape[0]):
                r = dist(grid[j, :], self.coordinates[i, :])
                if r < radius_scale * 0.99 * vdw_radii.get(self.numbers[i]):
                    not_near_atom[j] = False
        grid = grid[not_near_atom]

        # Double loop over grid to remove close lying points
        not_overlapping = np.ones(grid.shape[0], dtype=bool)
        for i in range(grid.shape[0]):
            for j in range(i + 1, grid.shape[0]):
                if (not not_overlapping[j]): continue  #already marked for removal
                r = dist(grid[i, :], grid[j, :])
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
        self.grid = np.concatenate(surfaces)
        self.ngridpoints = len(self.grid)
        self._T0 = None
        self._T1 = None
        self._T2 = None

    def compute_qm_esp(self):
        esp_grid_qm = self.obasis.compute_grid_esp_dm(self.dm, self.coordinates, self.numbers.astype(float),
                                                      self.grid)
        self.esp_grid_qm = esp_grid_qm

    def compute_all(self):
        self.compute_grid()
        self.compute_qm_esp()

    def write_xyz(self, filename):
        with open(filename, "w") as f:
            f.write("{}\n\n".format(self.natoms))
            for i in range(self.natoms):
                atomname = periodictable.to_symbol[self.numbers[i]]
                f.write("{} {: .10f}   {: .10f}   {: .10f}\n".format(
                    atomname, self.coordinates[i, 0] * constants.bohr2angstrom,
                    self.coordinates[i, 1] * constants.bohr2angstrom,
                    self.coordinates[i, 2] * constants.bohr2angstrom))

    def write_grid(self, filename):
        with open(filename, "w") as f:
            f.write("{}\n\n".format(self.ngridpoints))
            for i in range(self.ngridpoints):
                atomname = 'H'
                f.write("{} {: .10f}   {: .10f}   {: .10f}\n".format(atomname,
                                                                     self.grid[i, 0] * constants.bohr2angstrom,
                                                                     self.grid[i, 1] * constants.bohr2angstrom,
                                                                     self.grid[i, 2] * constants.bohr2angstrom))

    def save_h5(self, filename):
        f = h5py.File(filename, "w")
        f.create_dataset("atom_coordinates", data=self.coordinates)
        f.create_dataset("atom_numbers", data=self.numbers)
        f.create_dataset("electric_potential", data=self.esp_grid_qm)
        f.create_dataset("grid_coordinates", data=self.grid)
        f.create_dataset("external_field", data=self.field)
        f.close()

    def load_h5(self, filename):
        f = h5py.File(filename, "r")
        self.coordinates = f["atom_coordinates"][()]
        self.numbers = f["atom_numbers"][()]
        self.esp_grid_qm = f["electric_potential"][()]
        self.grid = f["grid_coordinates"][()]
        self.natoms = len(self.numbers)
        if "external_field" in f.keys():
            self.field = f["external_field"][()]
        else:
            self.field = np.zeros(3, dtype=np.float64)
        f.close()

    @property
    def T0(self):
        if self._T0 is not None:
            return self._T0
        self._T0 = T0(self.coordinates, self.grid)
        return self._T0

    @property
    def T1(self):
        if self._T1 is not None:
            return self._T1
        self._T1 = T1(self.coordinates, self.grid)
        return self._T1

    @property
    def T2(self):
        if self._T2 is not None:
            return self._T2
        self._T2 = T2(self.coordinates, self.grid)
        return self._T2

    def get_rotation_matrices(self, constraints):
        self.rotation_matrices = np.zeros((self.natoms, 3, 3), dtype=np.float64)
        for idx in range(self.natoms):
            points = [self.coordinates[i, :] for i in [idx, *constraints.axis_atomindices[idx]]]
            self.rotation_matrices[idx, :, :] = getattr(rotations, constraints.axis_types[idx])(*points)


class fragment(object):
    def __init__(self, fragdict):
        self.atomindices = np.array(fragdict["atomindices"], dtype=np.int64) - 1
        self.atomnames = fragdict["atomnames"]
        self.qtot = fragdict["qtot"]
        self.symmetries = [list(np.array(x, dtype=np.int64) - 1) for x in fragdict["symmetries"]]
        self.fullsymmetries = []
        self.natoms = len(self.atomindices)
        self.symmetryidx = np.copy(self.atomindices)
        self.nparamtersq = 0
        self.nparamtersa = 0
        self.lastidx = self.atomindices[-1]
        self.lastidxissym = False
        self.lastidxnsym = 1  #standard, no symmetry on last atom
        self.lastidxsym = [self.lastidx]
        self.startguess_charge = fragdict["startguess_charge"]
        self.startguess_dipole = fragdict["startguess_dipole"]
        self.startguess_quadrupole = fragdict["startguess_quadrupole"]
        self.startguess_polarizability = fragdict["startguess_polarizability"]
        self.axis_types = fragdict["axis_types"]
        self.axis_atomindices = [[idx - 1 for idx in axis_atomindices]
                                 for axis_atomindices in fragdict["axis_atomindices"]]
        self.axis_atomnames = fragdict["axis_atomnames"]
        self.axis_number_of_symmetric = fragdict["axis_number_of_symmetric"]

        for iloc, idx in enumerate(self.symmetryidx):
            for sym in self.symmetries:
                if idx in sym:
                    self.symmetryidx[iloc] = sym[0]
                    if idx == self.lastidx:
                        self.lastidxissym = True
                        self.lastidxsym = sym
                        self.lastidxnsym = len(sym)

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
        self.filename = filename
        self.name = data["name"]
        self.restraint = 0.0
        self.fragments = []
        for fragdict in data["fragments"]:
            self.fragments.append(fragment(fragdict))
        self.natoms = 0

        self.startguess_charge_redundant = []
        self.startguess_dipole_redundant = []
        self.startguess_quadrupole_redundant = []
        self.atomnames = []
        self.axis_types = []
        self.axis_atomindices = []
        self.axis_atomnames = []
        self.axis_number_of_symmetric = []
        self.qtot = 0.
        for frag in self.fragments:
            self.natoms += frag.natoms
            self.qtot += frag.qtot
            self.atomnames += frag.atomnames
            self.startguess_charge_redundant += frag.startguess_charge
            self.startguess_dipole_redundant += frag.startguess_dipole
            self.startguess_quadrupole_redundant += frag.startguess_quadrupole
            self.axis_types += frag.axis_types
            self.axis_atomindices += list(frag.axis_atomindices)
            self.axis_atomnames += frag.axis_atomnames
            self.axis_number_of_symmetric += frag.axis_number_of_symmetric
        self.startguess_charge_redundant = np.array(self.startguess_charge_redundant)
        self.startguess_dipole_redundant = np.array(self.startguess_dipole_redundant)
        self.startguess_quadrupole_redundant = np.array(self.startguess_quadrupole_redundant)

        #symmetrize start-guesses
        for frag in self.fragments:
            for sym in frag.fullsymmetries:
                q_sym = 0.0
                for member in sym:
                    q_sym += self.startguess_charge_redundant[member]
                q_sym = q_sym / len(sym)
                for member in sym:
                    self.startguess_charge_redundant[member] = q_sym
            excess = 0.
            for index in frag.atomindices:
                excess += self.startguess_charge_redundant[index]
            excess = excess - frag.qtot
            for index in frag.fullsymmetries[-1]:
                self.startguess_charge_redundant[index] -= excess / len(frag.fullsymmetries[-1])
        for frag in self.fragments:
            for sym in frag.fullsymmetries:
                dipole_sym = np.zeros(3)
                quadrupole_sym = np.zeros((3, 3))
                for member in sym:
                    dipole_sym += self.startguess_dipole_redundant[member, :]
                    quadrupole_sym += self.startguess_quadrupole_redundant[member, :, :]
                dipole_sym /= len(sym)
                quadrupole_sym /= len(sym)
                for member in sym:
                    self.startguess_dipole_redundant[member, :] = dipole_sym
                    self.startguess_quadrupole_redundant[member, :, :] = quadrupole_sym

    def get_multipole_parameter_vector(self,
                                       optimize_charges=True,
                                       optimize_dipoles=True,
                                       optimize_quadrupoles=True,
                                       dipole_zero_threshold=1e-2,
                                       quadrupole_zero_threshold=1e-2):
        self.optimize_charges = optimize_charges
        self.optimize_dipoles = optimize_dipoles
        self.optimize_quadrupoles = optimize_quadrupoles
        parameter_vector = []
        q0 = []
        mu0 = []
        theta0 = []
        self.nparametersq = 0
        self.nparametersmu = 0
        self.nparameterstheta = 0
        if optimize_charges:
            # read redundant charges
            q_red = []
            for frag in self.fragments:
                self.qtot += frag.qtot
                self.nparametersq += frag.nparametersq
                q_red += frag.startguess_charge

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
            q0 = [q_red[index] for index in indices]
            np.zeros(self.nparametersq, dtype=np.float64)

        if optimize_dipoles:
            self.dipole_parameters_active = []
            for frag in self.fragments:
                for sym in frag.fullsymmetries:
                    dipole = np.zeros(3)
                    for index in sym:
                        dipole += np.array(self.startguess_dipole_redundant[index])
                    dipole = dipole / len(sym)
                    # check which parameters are non-zero
                    # "bool x -> is_nonzero(x)"
                    x, y, z = dipole_axis_nonzero = np.abs(dipole) > dipole_zero_threshold
                    if x: mu0.append(dipole[0])
                    if y: mu0.append(dipole[1])
                    if z: mu0.append(dipole[2])
                    self.dipole_parameters_active.append([x, y, z])

        if optimize_quadrupoles:
            self.quadrupole_parameters_active = []
            for frag in self.fragments:
                for sym in frag.fullsymmetries:
                    quadrupole = np.zeros((3, 3))
                    for index in sym:
                        quadrupole += np.array(self.startguess_dipole_redundant[index])
                    quadrupole = quadrupole / len(sym)
                    # check which parameters are non-zero by local symmetry
                    # "bool xy -> is_nonzero(xy)"
                    xx = np.abs(quadrupole[0, 0]) > quadrupole_zero_threshold
                    xy = np.abs(quadrupole[0, 1]) < quadrupole_zero_threshold
                    xz = np.abs(quadrupole[0, 2]) < quadrupole_zero_threshold
                    yy = np.abs(quadrupole[1, 1]) < quadrupole_zero_threshold
                    yz = np.abs(quadrupole[1, 2]) < quadrupole_zero_threshold
                    if xx: theta0.append(quadrupole[0, 0])
                    if xy: theta0.append(quadrupole[0, 1])
                    if xz: theta0.append(quadrupole[0, 2])
                    if yy: theta0.append(quadrupole[1, 1])
                    if yz: theta0.append(quadrupole[1, 2])
                    self.quadrupole_parameters_active.append([xx, xy, xz, yy, yz])
        self.nparametersmu = len(mu0)
        self.nparameterstheta = len(theta0)
        parameter_vector = q0 + mu0 + theta0
        return parameter_vector

    def expand_parameter_vector(self, parameter_vector):
        pcounter = 0
        if self.optimize_charges:
            charges = np.zeros(self.natoms, dtype=np.float64)
            for frag in self.fragments:
                qcur = 0.0
                for sym in frag.fullsymmetries[:-1]:
                    for idx in sym:
                        charges[idx] = parameter_vector[pcounter]
                        qcur += charges[idx]
                    pcounter += 1
                #charge constraint. lastidxnsym is 1 if the last one is not a part of a symmetry
                qlast = (frag.qtot - qcur) / len(frag.fullsymmetries[-1])
                for idx in frag.fullsymmetries[-1]:
                    charges[idx] = qlast
        else:
            charges = self.startguess_charge_redundant
        if self.optimize_dipoles:
            dipoles = np.zeros((self.natoms, 3))
            dipole_pcounter = 0
            for frag in self.fragments:
                for sym in frag.fullsymmetries:
                    x, y, z = self.dipole_parameters_active[dipole_pcounter]
                    dipole_pcounter += 1
                    mx = my = mz = 0.
                    if x:
                        mx = parameter_vector[pcounter]
                        pcounter += 1
                    if y:
                        my = parameter_vector[pcounter]
                        pcounter += 1
                    if z:
                        mz = parameter_vector[pcounter]
                        pcounter += 1
                    for idx in sym:
                        dipoles[idx, 0] = mx
                        dipoles[idx, 1] = my
                        dipoles[idx, 2] = mz
        else:
            dipoles = self.startguess_dipole_redundant
        if self.optimize_quadrupoles:
            quadrupoles = np.zeros((self.natoms, 3, 3))
            quadrupole_pcounter = 0
            for frag in self.fragments:
                for sym in frag.fullsymmetries:
                    xx, xy, xz, yy, yz = self.quadrupole_parameters_active[quadrupole_pcounter]
                    quadrupole_pcounter += 1
                    Qxx = Qxy = Qxz = Qyy = Qyz = Qzz = 0.
                    if xx:
                        Qxx = parameter_vector[pcounter]
                        pcounter += 1
                    if xy:
                        Qxy = parameter_vector[pcounter]
                        pcounter += 1
                    if xz:
                        Qxz = parameter_vector[pcounter]
                        pcounter += 1
                    if yy:
                        Qyy = parameter_vector[pcounter]
                        pcounter += 1
                    if yz:
                        Qyz = parameter_vector[pcounter]
                        pcounter += 1
                    for idx in sym:
                        quadrupoles[idx, 0, 0] = Qxx
                        quadrupoles[idx, 0, 1] = Qxy
                        quadrupoles[idx, 1, 0] = Qxy
                        quadrupoles[idx, 0, 2] = Qxz
                        quadrupoles[idx, 2, 0] = Qxz
                        quadrupoles[idx, 1, 1] = Qyy
                        quadrupoles[idx, 1, 2] = Qyz
                        quadrupoles[idx, 2, 1] = Qyz
                        quadrupoles[idx, 2, 2] = -(Qxx + Qyy)
        else:
            quadrupoles = self.startguess_quadrupole_redundant
        return charges, dipoles, quadrupoles

    def expand_charges_for_fdiff(self, parameter_vector):
        # for difference charge, qtot = 0
        qtot = 0.
        charges = np.zeros(self.natoms, dtype=np.float64)
        pcounter = 0
        for frag in self.fragments:
            qcur = 0.0
            for sym in frag.fullsymmetries[:-1]:
                for idx in sym:
                    charges[idx] = parameter_vector[pcounter]
                    qcur += charges[idx]
                pcounter += 1
            #charge constraint. lastidxnsym is 1 if the last one is not a part of a symmetry
            qlast = (qtot - qcur) / len(frag.fullsymmetries[-1])
            for idx in frag.fullsymmetries[-1]:
                charges[idx] = qlast
        return charges

    def expand_dipoles_for_fdiff(self, parameter_vector):
        pcounter = self.nparametersq
        dipoles = np.zeros((self.natoms, 3))
        dipole_pcounter = 0
        for frag in self.fragments:
            for sym in frag.fullsymmetries:
                x, y, z = self.dipole_parameters_active[dipole_pcounter]
                dipole_pcounter += 1
                mx = my = mz = 0.
                if x:
                    mx = parameter_vector[pcounter]
                    pcounter += 1
                if y:
                    my = parameter_vector[pcounter]
                    pcounter += 1
                if z:
                    mz = parameter_vector[pcounter]
                    pcounter += 1
                for idx in sym:
                    dipoles[idx, 0] = mx
                    dipoles[idx, 1] = my
                    dipoles[idx, 2] = mz
        return dipoles

    def expand_quadrupoles_for_fdiff(self, parameter_vector):
        pcounter = self.nparametersq + self.nparametersmu
        quadrupoles = np.zeros((self.natoms, 3, 3))
        quadrupole_pcounter = 0
        for frag in self.fragments:
            for sym in frag.fullsymmetries:
                xx, xy, xz, yy, yz = self.quadrupole_parameters_active[quadrupole_pcounter]
                quadrupole_pcounter += 1
                Qxx = Qxy = Qxz = Qyy = Qyz = Qzz = 0.
                if xx:
                    Qxx = parameter_vector[pcounter]
                    pcounter += 1
                if xy:
                    Qxy = parameter_vector[pcounter]
                    pcounter += 1
                if xz:
                    Qxz = parameter_vector[pcounter]
                    pcounter += 1
                if yy:
                    Qyy = parameter_vector[pcounter]
                    pcounter += 1
                if yz:
                    Qyz = parameter_vector[pcounter]
                    pcounter += 1
                for idx in sym:
                    quadrupoles[idx, 0, 0] = Qxx
                    quadrupoles[idx, 0, 1] = Qxy
                    quadrupoles[idx, 1, 0] = Qxy
                    quadrupoles[idx, 0, 2] = Qxz
                    quadrupoles[idx, 2, 0] = Qxz
                    quadrupoles[idx, 1, 1] = Qyy
                    quadrupoles[idx, 1, 2] = Qyz
                    quadrupoles[idx, 2, 1] = Qyz
                    quadrupoles[idx, 2, 2] = -(Qxx + Qyy)
        return quadrupoles
