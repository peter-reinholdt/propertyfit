import numpy as np


def induced_dipole(alpha_ab, field):
    """Anisotropic atom-centered dipole-dipole polarizabilities in a homogenous field"""
    natoms = len(alpha_ab)
    mu = np.zeros((natoms,3))
    for i in range(natoms):
        for j in range(3):
            mu[i,j] += alpha_ab[i,j] * field[j]
    return mu


def dipole_potential(dipoles, rinvmat, xyzmat):
    natoms      = rinvmat.shape[0] 
    ngridpoints = rinvmat.shape[1]
    potential = np.zeros(ngridpoints)
    for i in range(natoms):
        for j in range(ngridpoints):
            for k in range(3):
                potential[j] -= dipoles[i,k] * rinvmat[i,j]**3 * xyzmat[i,j,k]
    return potential
