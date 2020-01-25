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
from .potentials import field


@jit(nopython=True)
def charge_esp_square_error(rinvmat, esp_grid_qm, testcharges):
    """Compute the average square error
    between the ESP set up by points charge
    and the full QM ESP. 
    The error is evaluated in the grid points
    and the average of the squares of the errors
    is returned
    """
    natoms = rinvmat.shape[0]
    ngridpoints = rinvmat.shape[1]
    grid = np.copy(esp_grid_qm)
    for i in range(natoms):
        for j in range(ngridpoints):
            grid[j] -= testcharges[i] * rinvmat[i, j]
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
    mu = np.zeros((natoms, 3))
    for i in range(natoms):
        for j in range(3):
            for k in range(3):
                #mu_i,alpha = alpha_i,alphabeta*Fbeta
                mu[i, j] += alpha_ab[i, j, k] * field[k]
    return mu


@jit(nopython=True)
def dipole_potential(dipoles, rinvmat, xyzmat):
    """Compute the potential from a set of
    dipoles in the defined gridpoints"""
    natoms = rinvmat.shape[0]
    ngridpoints = rinvmat.shape[1]
    potential = np.zeros(ngridpoints)
    for i in range(natoms):
        for j in range(ngridpoints):
            for k in range(3):
                potential[j] -= dipoles[i, k] * rinvmat[i, j]**3 * xyzmat[i, j, k]
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
    natoms = rinvmat.shape[0]
    ngridpoints = rinvmat.shape[1]
    mu_ind = induced_dipole(alpha_ab, field)
    alpha_pot = dipole_potential(mu_ind, rinvmat, xyzmat)
    return np.sum((induced_esp_grid_qm - alpha_pot)**2) / ngridpoints


def charge_cost_function(qtest, structures=None, constraints=None, filter_outliers=True, weights=None):
    """
    Cost function for charges, based on the average of 
    charge_esp_square_error across all structures.

    qtest:          array of non-redundant test-charge parameters
    structures:     list of structure objects
    constraints:    constraints object, which contains information
                    about symmetries etc.
    """
    #expand charges to full set
    qfull = constraints.expand_q(qtest)
    qfull_ref = constraints.expand_q(constraints.q0)
    nstructures = len(structures)
    res = 0.0

    if weights is not None:
        #make sure it is normalized
        weights = weights / np.sum(weights)
    else:
        weights = np.zeros(nstructures)
        weights[:] = 1.0 / nstructures

    contributions = np.zeros(nstructures)
    for i, s in enumerate(structures):
        contribution = charge_esp_square_error(s.rinvmat, s.esp_grid_qm, qfull)
        contributions[i] = contribution * weights[i]
    if filter_outliers:
        median = np.median(contributions)
        filtered = contributions > median * 100.
        res = np.sum(contributions[~filtered])
    else:
        res = np.sum(contribution)

    #print the pure version of the cost function
    #print(res)

    #restraints towards zero for increased stability
    if constraints.restraint > 0.0:
        for i in range(constraints.natoms):
            res += constraints.restraint * (qfull[i] - qfull_ref[i])**2
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
    afull = constraints.expand_a(alphatest)
    afull_ref = constraints.expand_a(constraints.a0)
    nstructures = len(structures)

    if weights is not None:
        #make sure it is normalized
        weights = weights / np.sum(weights)
    else:
        weights = np.zeros(nstructures)
        weights[:] = 1.0 / nstructures

    res = 0.0
    for i in range(nstructures):
        contribution = induced_esp_sum_squared_error(structures[i].rinvmat, structures[i].xyzmat,
                                                     structures[i].esp_grid_qm - fieldstructures[i].esp_grid_qm,
                                                     fieldstructures[i].field, afull)
        res += contribution * weights[i]
    #print(res)
    if constraints.restraint > 0.0:
        for i in range(constraints.natoms):
            res += constraints.restraint * (afull[i][0, 0] - afull_ref[i][0, 0])**2
    return res


def multipole_restraint_contribution_res(parameters, constraints):
    charge_parameters = parameters[0:constraints.nparametersq]
    dipole_parameters = parameters[constraints.nparametersq:constraints.nparametersq + constraints.nparametersmu]
    quadrupole_parameters = parameters[constraints.nparametersq + constraints.nparametersmu:constraints.nparametersq +
                                       constraints.nparametersmu + constraints.nparameterstheta]

    charges = constraints.expand_charges(charge_parameters)
    ref_charges = constraints.expand_charges(constraints.q0)
    dipoles_local = constraints.expand_dipoles(dipole_parameters)
    quadrupoles_local = constraints.expand_quadrupoles(quadrupole_parameters)
    ref_dipoles = constraints.expand_dipoles(constraints.mu0)
    ref_quadrupoles = constraints.expand_quadrupoles(constraints.theta0)
    # charge
    res = np.sum((charges - ref_charges)**2)
    res += np.sum((dipoles_local - ref_dipoles)**2)
    res += np.sum((quadrupoles_local - ref_quadrupoles)**2)
    res *= constraints.restraint
    return res


def multipole_restraint_contribution_jac(parameters, constraints):
    h = 1e-6
    jac = np.zeros(parameters.shape)
    for ip in range(len(parameters)):
        pplus = np.copy(parameters)
        pplus[ip] += h
        pminus = np.copy(parameters)
        pminus[ip] -= h
        jac[ip] = (multipole_restraint_contribution_res(pplus, constraints) -
                   multipole_restraint_contribution_res(pminus, constraints)) / (2 * h)
    return jac


def multipole_cost_function(parameters, structures=None, constraints=None, filter_outliers=True, weights=None):
    """
    Cost function for multipoles, based on the average of 
    square esp error across all structures.

    parameters:     array of non-redundant test-charge parameters
                    parameters[0:nparametersq] -> charges
                    parameters[nparametersq:nparametersq+nparametersmu] -> dipole parameters
                    parameters[nparametersq+nparametersmu:nparametersq+nparametersmu+nparamererstheta] -> quadrupole parameters
    structures:     list of structure objects
    constraints:    constraints object, which contains information
                    about symmetries etc.
    """
    #expand multipole parameters to full set (and rotate dipole, quadrupole from local axis to global axis)
    charge_parameters = parameters[0:constraints.nparametersq]
    dipole_parameters = parameters[constraints.nparametersq:constraints.nparametersq + constraints.nparametersmu]
    quadrupole_parameters = parameters[constraints.nparametersq + constraints.nparametersmu:constraints.nparametersq +
                                       constraints.nparametersmu + constraints.nparameterstheta]

    charges = constraints.expand_charges(charge_parameters)
    dipoles_local = constraints.expand_dipoles(dipole_parameters)
    quadrupoles_local = constraints.expand_quadrupoles(quadrupole_parameters)

    nstructures = len(structures)
    res = 0.0

    if weights is not None:
        #make sure it is normalized
        weights = weights / np.sum(weights)
    else:
        weights = np.zeros(nstructures)
        weights[:] = 1.0 / nstructures

    contributions = np.zeros(nstructures)
    diff_esps = []
    for idx, s in enumerate(structures):
        test_esp = np.zeros(s.esp_grid_qm.shape)
        #minus sign due to potential definition
        dipoles, quadrupoles = constraints.rotate_multipoles_to_global_axis(dipoles_local, quadrupoles_local, s)
        test_esp += -field(s, 0, charges, 0, idx)
        test_esp += -field(s, 1, dipoles, 0, idx)
        test_esp += -field(s, 2, quadrupoles, 0, idx)
        diff_esps.append(test_esp - s.esp_grid_qm)
        contribution = np.average((test_esp - s.esp_grid_qm)**2)
        contributions[idx] = contribution * weights[idx]
    res = np.sum(contributions)

    # get jacobian
    jac = np.zeros(parameters.shape)
    h = 1e-6
    test_charge = np.zeros(constraints.nparametersq)
    test_dipole = np.zeros(constraints.nparametersmu)
    test_quadrupole = np.zeros(constraints.nparameterstheta)

    # EP contribution
    for ip in range(len(parameters)):
        if ip < constraints.nparametersq:
            test_charge[:] = 0.
            test_charge[ip] = h
            charges = constraints.expand_charges_for_fdiff(test_charge)
            mask = charges != 0.
            j = 0.0
            for idx, s in enumerate(structures):
                esp = -field(s.coordinates[mask, :], s.grid, 0, charges[mask], 0, (ip, idx))
                j += weights[idx] * (np.average((diff_esps[idx] + esp)**2) - np.average((diff_esps[idx] - esp)**2))
            jac[ip] = j / (2 * h)
        elif ip < constraints.nparametersq + constraints.nparametersmu:
            test_dipole[:] = 0.
            test_dipole[ip - constraints.nparametersq] = h
            dipoles_local = constraints.expand_dipoles(test_dipole)
            mask = np.any(dipoles_local != 0., axis=1)
            j = 0.0
            for idx, s in enumerate(structures):
                dipoles = constraints.rotate_dipoles_to_global_axis(dipoles_local, s)
                esp = -field(s.coordinates[mask, :], s.grid, 1, dipoles[mask, :], 0, (ip, idx))
                j += weights[idx] * (np.average((diff_esps[idx] + esp)**2) - np.average((diff_esps[idx] - esp)**2))
            jac[ip] = j / (2 * h)
        elif ip < constraints.nparametersq + constraints.nparametersmu + constraints.nparameterstheta:
            test_quadrupole[:] = 0.
            test_quadrupole[ip - constraints.nparametersq - constraints.nparametersmu] = h
            quadrupoles_local = constraints.expand_quadrupoles(test_quadrupole)
            mask = np.any(quadrupoles_local != 0., axis=(1, 2))
            j = 0.0
            for idx, s in enumerate(structures):
                quadrupoles = constraints.rotate_quadrupoles_to_global_axis(quadrupoles_local, s)
                esp = -field(s.coordinates[mask, :], s.grid, 2, quadrupoles[mask, :, :], 0, (ip, idx))
                j += weights[idx] * (np.average((diff_esps[idx] + esp)**2) - np.average((diff_esps[idx] - esp)**2))
            jac[ip] = j / (2 * h)

    # restraint contribution
    if constraints.restraint > 0.:
        res_restraint = multipole_restraint_contribution_res(parameters, constraints)
        jac_restraint = multipole_restraint_contribution_jac(parameters, constraints)
        res += res_restraint
        jac += jac_restraint

    # scale by large number to make optimizer work better...
    # or "units in (mH)**2"
    return 1e6 * res, 1e6 * jac
