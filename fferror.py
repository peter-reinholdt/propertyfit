#!/usr/bin/env python
import sys
import numpy as np
from numba import jit
import writePDB
from propertyfit.utilities import load_qmfiles
from propertyfit.utilities import angstrom2bohr, hartree2kjmol, number2name


@jit(nopython=True, cache=True)
def charge_esp(rinvmat, testcharges):
    natoms      = rinvmat.shape[0]
    ngridpoints = rinvmat.shape[1]
    grid = np.zeros(ngridpoints, dtype=np.float64)
    for i in range(natoms):
        for j in range(ngridpoints):
            grid[j] += testcharges[i] * rinvmat[i,j]
    return grid


@jit(nopython=True, cache=True)
def dipole_esp(rinvmat, xyzmat, testdipoles):
    natoms      = rinvmat.shape[0] 
    ngridpoints = rinvmat.shape[1]
    grid = np.zeros(ngridpoints, dtype=np.float64)
    for i in range(natoms):
        for j in range(ngridpoints):
            for k in range(3):
                grid[j] -= testdipoles[i,k] * rinvmat[i,j]**3 * xyzmat[i,j,k]
    return grid

if __name__ == "__main__":
    qmfile     = sys.argv[1]
    chargefile = sys.argv[2]
    dipolefile = sys.argv[3]
    print(qmfile, chargefile, dipolefile)

    testcharges = np.loadtxt(chargefile)
    testdipoles = np.loadtxt(dipolefile)

    structure = load_qmfiles(qmfile)[0]

    #PDB out stuff
    atom_coords = structure.coordinates
    numbers     = structure.numbers
    labels      = [number2name[num] for num in numbers]


    print("Computing grid...")
    structure.compute_grid(rmin=2.0*angstrom2bohr, rmax=2.0*angstrom2bohr, nsurfaces=1, pointdensity=1.0)
    print("Computing rinvmat...")
    structure.compute_rinvmat()
    print("Computing xyzmat...")
    structure.compute_xyzmat()
    print("Computing QM ESP...")
    structure.compute_qm_esp()


    V = np.zeros((structure.grid.shape[0],5))
    V[:,0:3] = structure.grid
    V[:,  3] = structure.esp_grid_qm


    rinvmat = np.copy(structure.rinvmat)
    xyzmat  = np.copy(structure.xyzmat)

    #subtract charge contribution
    structure.esp_grid_qm -= charge_esp(rinvmat, testcharges)
    V[:,4] = structure.esp_grid_qm
    writePDB.writePDB(V, labels, atom_coords, "{}_charge.pdb".format(qmfile))

    RMSD = np.sqrt(np.average(structure.esp_grid_qm**2))
    print("Charges: RMSD of {} is {} kJ/mol".format(qmfile, RMSD*hartree2kjmol))


    #subtract dipole contribution
    structure.esp_grid_qm -= dipole_esp(rinvmat, xyzmat, testdipoles)
    V[:,4] = structure.esp_grid_qm
    writePDB.writePDB(V, labels, atom_coords, "{}_charge+alpha.pdb".format(qmfile))

    RMSD = np.sqrt(np.average(structure.esp_grid_qm**2))
    print("Charges + Dipoles: RMSD of {} is {} kJ/mol".format(qmfile, RMSD*hartree2kjmol))

