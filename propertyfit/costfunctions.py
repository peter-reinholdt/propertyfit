#!/usr/bin/env python
"""
Contains routines for evaluating potential from
test charges and dipoles, as well as definitions
of cost functions for fitting charges and
(isotropic) polarizabilities
"""

from __future__ import print_function
import numpy as np
from .utilities import hartree2kjmol
from .potentials import field

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
    charges, dipoles_local, quadrupoles_local = constraints.expand_parameter_vector(parameters)
    res = np.sum((charges - constraints.startguess_charge_redundant)**2)
    res += np.sum((dipoles_local - constraints.startguess_dipole_redundant)**2)
    res += np.sum((quadrupoles_local - constraints.startguess_quadrupole_redundant)**2)
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
    charges, dipoles_local, quadrupoles_local = constraints.expand_parameter_vector(parameters)

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
        # minus sign due to potential definition
        # R @ mu
        dipoles = np.einsum("aij,aj->ai", s.rotation_matrices, dipoles_local)
        # R @ theta @ R.T
        quadrupoles = np.einsum("aij,ajk,alk->ail", s.rotation_matrices, quadrupoles_local, s.rotation_matrices)
        test_esp += -field(s, 0, charges, 0)
        test_esp += -field(s, 1, dipoles, 0)
        test_esp += -field(s, 2, quadrupoles, 0)
        diff_esps.append(test_esp - s.esp_grid_qm)
        contribution = np.average((test_esp - s.esp_grid_qm)**2)
        contributions[idx] = contribution * weights[idx]
    res = np.sum(contributions)

    # get jacobian
    jac = np.zeros(parameters.shape)
    h = 1e-6
    test_parameter = np.zeros(parameters.shape[0])

    # EP contribution
    for ip in range(len(parameters)):
        if ip < constraints.nparametersq:
            test_parameter[:] = 0.
            test_parameter[ip] = h
            charges = constraints.expand_charges_for_fdiff(test_parameter)
            mask = charges != 0.
            j = 0.0
            for idx, s in enumerate(structures):
                esp = -field(s, 0, charges, 0, mask=mask)
                j += weights[idx] * (np.average((diff_esps[idx] + esp)**2) - np.average((diff_esps[idx] - esp)**2))
            jac[ip] = j / (2 * h)
        elif ip < constraints.nparametersq + constraints.nparametersmu:
            test_parameter[:] = 0.
            test_parameter[ip] = h
            dipoles_local = constraints.expand_dipoles_for_fdiff(test_parameter)
            mask = np.any(dipoles_local != 0., axis=1)
            j = 0.0
            for idx, s in enumerate(structures):
                dipoles = np.einsum("aij,aj->ai", s.rotation_matrices, dipoles_local)
                esp = -field(s, 1, dipoles, 0, mask=mask)
                j += weights[idx] * (np.average((diff_esps[idx] + esp)**2) - np.average((diff_esps[idx] - esp)**2))
            jac[ip] = j / (2 * h)
        elif ip < constraints.nparametersq + constraints.nparametersmu + constraints.nparameterstheta:
            test_parameter[:] = 0.
            test_parameter[ip] = h
            quadrupoles_local = constraints.expand_quadrupoles_for_fdiff(test_parameter)
            mask = np.any(quadrupoles_local != 0., axis=(1, 2))
            j = 0.0
            for idx, s in enumerate(structures):
                quadrupoles = np.einsum("aij,ajk,alk->ail", s.rotation_matrices, quadrupoles_local,
                                        s.rotation_matrices)
                esp = -field(s, 2, quadrupoles, 0, mask=mask)
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
