#!/usr/bin/env python
"""
Contains routines for evaluating potential from
test charges and dipoles, as well as definitions
of cost functions for fitting charges and
(isotropic) polarizabilities
"""

from __future__ import print_function
import numpy as np
from numba import jit
from .utilities import hartree2kjmol


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
    mu_ind      = induced_dipole(alpha_ab, field)
    alpha_pot   = dipole_potential(mu_ind, rinvmat, xyzmat)
    return np.sum((induced_esp_grid_qm-alpha_pot)**2) / ngridpoints


def charge_cost_function(qtest, structures=None, constraints=None, filter_outliers=True):
    """
    Cost function for charges, based on the average of 
    charge_esp_square_error across all structures.

    qtest:          array of non-redundant test-charge parameters
    structures:     list of structure objects
    constraints:    constraints object, which contains information
                    about symmetries etc.
    """
    #expand charges to full set
    qfull       = constraints.expand_q(qtest)
    qfull_ref   = constraints.expand_q(constraints.q0)
    nstructures = len(structures)
    res = 0.0

    if weights == None:
        weights = np.zeros(nstructures)
        weights[:] = 1.0/nstructures
    else:
        #make sure it is normalized
        weights = weights / np.sum(weights)
    contributions = np.zeros(nstructures)
    for i, s in enumerate(structures):
        contribution = charge_esp_square_error(s.rinvmat, s.esp_grid_qm, qfull)
        contributions[i] = contribution * weights[i]
    if filter_outliers:
        median = np.median(contributions)
        res = np.sum(contributions[contributions < median * 100.0])
        filtered = contributions > median * 100.
        if np.any(filtered):
            print('Contributions {} were {} times greater than the median contribution and were filtered.'.format([structures[i].fchkname for i in np.where(filtered)], contributions[filtered] /
                median), end =' ')
    else:
        res = np.sum(contribution)
    
    #print the pure version of the cost function
    print(res)
    
    #restraints towards zero for increased stability
    if constraints.restraint > 0.0:
        for i in range(constraints.natoms):
            res += constraints.restraint * (qfull[i]-qfull_ref[i])**2
    return res


def isopol_cost_function(alphatest, structures, fieldstructures, constraints, weights=None):
    """
    Cost function for isotropic polarizabilities, based on the average of 
    induced_esp_sum_squared_error across all structures.
    alphatest:      array of non-redundant test-polarizablity parameters
    structures:     list of structure objects
    constraints:    constraints object, which contains information
                    about symmetries etc.
    """
    afull       = constraints.expand_a(alphatest)
    afull_ref   = constraints.expand_a(constraints.a0)
    nstructures = len(structures)
    if weights == None:
        weights = np.zeros(nstructures)
        weights[:] = 1.0/nstructures
    else:
        #make sure it is normalized
        weights = weights / np.sum(weights)
    res         = 0.0
    for i in range(nstructures):
        contribution =  induced_esp_sum_squared_error(structures[i].rinvmat, 
                                             structures[i].xyzmat, 
                                             structures[i].esp_grid_qm - fieldstructures[i].esp_grid_qm, 
                                             fieldstructures[i].field, 
                                             afull)
        res += contribution * weights[i]
    print(res)
    if constraints.restraint > 0.0:
        for i in range(constraints.natoms):
            res += constraints.restraint * (afull[i][0,0]-afull_ref[i][0,0])**2
    return res
