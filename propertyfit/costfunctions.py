#!/usr/bin/env python
"""
Contains routines for evaluating potential from
test charges and dipoles, as well as definitions
of cost functions for fitting charges and
(isotropic) polarizabilities
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def charge_esp_square_error(rinvmat, esp_grid_qm, testcharges):
    """Compute the average square error
    between the ESP set up by points charge
    and the full QM ESP. 
    The error is evaluated in the grid points
    and the average of the squares of the errors
    is returned
    """
    natoms      = rinvmat.shape[0]
    ngridpoints = rinvmat.shape[1]
    grid = np.copy(esp_grid_qm)
    for i in range(natoms):
        for j in range(ngridpoints):
            grid[j] -= testcharges[i] * rinvmat[i,j]
    return np.sum(grid**2) / ngridpoints


@jit(nopython=True)
def induced_dipole(alpha_ab, field):
    """Calculate the set of induced dipoles
    set up by a homogenous field acting on 
    a set of polarizabilties
    The polarizabilities can be anisotropic
    and the full Nx3x3 polarizability tensor
    is used
    """
    natoms = len(alpha_ab)
    mu = np.zeros((natoms,3))
    for i in range(natoms):
        for j in range(3):
            for k in range(3):
                #mu_i,alpha = alpha_i,alphabeta*Fbeta
                mu[i,j] += alpha_ab[i,j,k] * field[k]
    return mu


@jit(nopython=True)
def dipole_potential(dipoles, rinvmat, xyzmat):
    """Compute the potential from a set of
    dipoles in the defined gridpoints"""
    natoms      = rinvmat.shape[0] 
    ngridpoints = rinvmat.shape[1]
    potential = np.zeros(ngridpoints)
    for i in range(natoms):
        for j in range(ngridpoints):
            for k in range(3):
                potential[j] -= dipoles[i,k] * rinvmat[i,j]**3 * xyzmat[i,j,k]
    return potential


@jit(nopython=True)
def induced_esp_sum_squared_error(rinvmat, xyzmat, induced_esp_grid_qm, field, alpha_ab):
    """Compute the average square error
    between the ESP set up by points dipole
    and the full QM induced ESP, where
    the induced ESP is defined as phi_QM(0) - phi_QM(F)
    The error is evaluated in the grid points
    and the average of the squares of the errors
    is returned
    """
    natoms      = rinvmat.shape[0]
    ngridpoints = rinvmat.shape[1]
    grid = np.copy(induced_esp_grid_qm)
    mu_ind = induced_dipole(alpha_ab, field)
    alpha_pot = dipole_potential(mu_ind, rinvmat, xyzmat)
    return np.sum((grid-alpha_pot)**2)


@jit(nopython=True)
def charge_cost_function(qtest, structures, constraints):
    """
    Cost function for charges, based on the average of 
    charge_esp_square_error across all structures.

    qtest:          array of non-redundant test-charge parameters
    structures:     list of structure objects
    constraints:    
    """
    return res


@jit(nopython=True)
def isopol_cost_function(alphatest, structures, constraints):
    """
    """
    return res
