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


def multipole_cost_function(parameters,
                            structures=None,
                            constraints=None,
                            filter_outliers=True,
                            weights=None,
                            calc_jac=False):
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
    jac = np.zeros(len(parameters))

    if weights is not None:
        #make sure it is normalized
        weights = weights / np.sum(weights)
    else:
        weights = np.zeros(nstructures)
        weights[:] = 1.0 / nstructures

    contributions = np.zeros(nstructures)
    diff_esps = []
    for idx, s in enumerate(structures):
        # minus sign due to potential definition
        test_esp = np.zeros(s.esp_grid_qm.shape)
        test_esp += -field(s, 0, charges, 0)
        if np.any(dipoles_local):
            # R @ mu
            dipoles = np.einsum("aij,aj->ai", s.rotation_matrices, dipoles_local)
            test_esp += -field(s, 1, dipoles, 0)
        if np.any(quadrupoles_local):
            # R @ theta @ R.T
            quadrupoles = np.einsum("aij,ajk,alk->ail", s.rotation_matrices, quadrupoles_local, s.rotation_matrices)
            test_esp += -field(s, 2, quadrupoles, 0)
        diff_esps.append(test_esp - s.esp_grid_qm)
        contribution = np.average((test_esp - s.esp_grid_qm)**2)
        contributions[idx] = contribution * weights[idx]
    res = np.sum(contributions)

    # restraint contribution
    if constraints.restraint:
        res_restraint = multipole_restraint_contribution_res(parameters, constraints)
        res += res_restraint

    # get jacobian
    if calc_jac:
        h = 1e-6
        test_parameter = np.zeros(len(parameters))

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
                    j += weights[idx] * (np.average((diff_esps[idx] + esp)**2) - np.average(
                        (diff_esps[idx] - esp)**2))
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
                    j += weights[idx] * (np.average((diff_esps[idx] + esp)**2) - np.average(
                        (diff_esps[idx] - esp)**2))
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
                    j += weights[idx] * (np.average((diff_esps[idx] + esp)**2) - np.average(
                        (diff_esps[idx] - esp)**2))
                jac[ip] = j / (2 * h)

        # restraint contribution
        if constraints.restraint:
            jac_restraint = multipole_restraint_contribution_jac(parameters, constraints)
            jac += jac_restraint

    # scale by large number to make optimizer work better...
    # or "units in (mH)**2"
    if calc_jac:
        return 1e6 * res, 1e6 * jac
    else:
        return 1e6 * res


def multipole_restraint_contribution_res(parameters, constraints):
    charges, dipoles_local, quadrupoles_local = constraints.expand_parameter_vector(parameters)
    res = constraints.restraint[0] * np.sum((charges - constraints.startguess_charge_redundant)**2)
    if np.any(dipoles_local):
        nonzero = dipoles_local != 0.
        res += constraints.restraint[1] * np.sum(
            (dipoles_local[nonzero] - constraints.startguess_dipole_redundant[nonzero])**2)
    if np.any(quadrupoles_local):
        nonzero = quadrupoles_local != 0.
        res += constraints.restraint[2] * np.sum(
            (quadrupoles_local[nonzero] - constraints.startguess_quadrupole_redundant[nonzero])**2)
    return res


def multipole_restraint_contribution_jac(parameters, constraints):
    h = 1e-6
    jac = np.zeros(len(parameters))
    for ip in range(len(parameters)):
        pplus = np.copy(parameters)
        pplus[ip] += h
        pminus = np.copy(parameters)
        pminus[ip] -= h
        jac[ip] = (multipole_restraint_contribution_res(pplus, constraints) -
                   multipole_restraint_contribution_res(pminus, constraints)) / (2 * h)
    return jac


def polarizability_cost_function(parameters, structures, fieldstructures, constraints, weights=None, calc_jac=False):
    # todo: implement
    polarizabilities_local = constraints.expand_polarizabilities(parameters)
    nstructures = len(structures)

    if weights is not None:
        weights = weights / np.sum(weights)
    else:
        weights = np.zeros(nstructures)
        weights[:] = 1.0 / nstructures

    res = 0.0
    jac = np.zeros(len(parameters))
    contributions = np.zeros(nstructures)
    diff_esps = []
    for idx, (fs, s) in enumerate(zip(fieldstructures, structures)):
        test_esp = np.zeros(s.esp_grid_qm.shape)
        # minus sign due to potential definition
        if constraints.isotropic_polarizabilities:
            # no need to rotate; they should be ~ np.eye(3) * aiso
            polarizabilities = polarizabilities_local
        else:
            # R @ alpha @ R.T
            polarizabilities = np.einsum("aij,ajk,alk->ail", fs.rotation_matrices, polarizabilities_local,
                                         fs.rotation_matrices)
        induced_dipoles = np.einsum("aij,j->ai", polarizabilities, fs.field)
        test_esp = -field(s, 1, induced_dipoles, 0)
        qm_induced_esp = s.esp_grid_qm - fs.esp_grid_qm
        esp_difference = test_esp - qm_induced_esp
        diff_esps.append(esp_difference)
        contribution = np.average((esp_difference)**2)
        contributions[idx] = contribution * weights[idx]
    res = np.sum(contributions)

    if calc_jac:
        h = 1e-6
        test_parameter = np.zeros(len(parameters))
        # EP contribution
        for ip in range(len(parameters)):
            test_parameter[:] = 0.
            test_parameter[ip] = h
            polarizabilities_local = constraints.expand_polarizabilities(test_parameter)
            mask = np.any(polarizabilities_local != 0., axis=(1, 2))
            j = 0.0
            for idx, fs in enumerate(fieldstructures):
                if constraints.isotropic_polarizabilities:
                    # no need to rotate; they should be ~ np.eye(3) * aiso
                    polarizabilities = polarizabilities_local
                else:
                    polarizabilities = np.einsum("aij,ajk,alk->ail", fs.rotation_matrices, polarizabilities_local,
                                                 fs.rotation_matrices)
                induced_dipoles = np.einsum("aij,j->ai", polarizabilities, fs.field)
                esp = -field(fs, 1, induced_dipoles, 0)
                j += weights[idx] * (np.average((diff_esps[idx] + esp)**2) - np.average((diff_esps[idx] - esp)**2))
            jac[ip] = j / (2 * h)

    # restraint contribution
    if constraints.restraint:
        res_restraint = polarizability_restraint_contribution_res(parameters, constraints)
        jac_restraint = polarizability_restraint_contribution_jac(parameters, constraints)
        res += res_restraint
        jac += jac_restraint

    # scale by large number to make optimizer work better...
    # or "units in (mH)**2"
    # print(res, np.linalg.norm(jac))
    if calc_jac:
        return 1e6 * res, 1e6 * jac
    else:
        return 1e6 * res


def polarizability_restraint_contribution_res(parameters, constraints):
    polarizabilities_local = constraints.expand_polarizabilities(parameters)
    nonzero = polarizabilities_local != 0.
    res = np.sum((polarizabilities_local[nonzero] - constraints.startguess_polarizability_redundant[nonzero])**2)
    res *= constraints.restraint
    return res


def polarizability_restraint_contribution_jac(parameters, constraints):
    h = 1e-6
    jac = np.zeros(len(parameters))
    for ip in range(len(parameters)):
        pplus = np.copy(parameters)
        pplus[ip] += h
        pminus = np.copy(parameters)
        pminus[ip] -= h
        # maybe it is 2*(a-aref) ?
        jac[ip] = (polarizability_restraint_contribution_res(pplus, constraints) -
                   polarizability_restraint_contribution_res(pminus, constraints)) / (2 * h)
    return jac
