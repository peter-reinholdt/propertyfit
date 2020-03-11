import os
import sys
import numpy as np
import functools
from scipy.optimize import minimize

this_file_location = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_file_location + '/../../')
from propertyfit.structures import structure, constraints
from propertyfit.costfunctions import multipole_cost_function, polarizability_cost_function


def test_run_charge():
    this_file_location = os.path.dirname(os.path.abspath(__file__))

    constraintsfile = this_file_location + '/../../constraints/VAL_methyl_methyl.constraints.new'
    files = [this_file_location + '/fchks_test/esp_VAL_methyl_methyl_0.h5']

    #create constraints object
    con = constraints(constraintsfile)

    #load structure objects as defined from locationfile
    structures = []
    for fname in files:
        s = structure()
        s.load_h5(fname)
        structures.append(s)

    #use partial to wrap cost function, so we only need a single qtest argument (and not constraints, structures)
    #then we can call fun(qtest) instead of charge_cost_function(qtest, structures, constraints)

    con.restraint = [2e-5, None, None]
    x0 = con.get_multipole_parameter_vector(optimize_charges=True, optimize_dipoles=False, optimize_quadrupoles=False)
    assert len(x0) == 15

    # test that we arrive at the same fit
    fun = functools.partial(multipole_cost_function, structures=structures, constraints=con)
    res = minimize(fun, x0=x0, method='SLSQP', tol=1e-12, options={'maxiter': 1000})
    q_ref = np.array([
        0.51992121, -0.46557429, -0.48780886, -0.38534308, 0.23874176, -0.06002675, 0.14176795, 0.00106602,
        0.56106577, 0.08579874, -0.43597247, -0.51824137, -0.33964451, 0.24573569, -0.31317867
    ])
    assert res.success
    assert np.allclose(res.x, q_ref, atol=1e-3)

    # test gradient implementation
    con.restraint = None
    grad_fdiff = minimize(fun, x0=x0, method='SLSQP', options={'maxiter': 0}).jac
    grad = functools.partial(multipole_cost_function, structures=structures, constraints=con, calc_jac=True)(x0)[1]
    assert np.allclose(grad_fdiff, grad)
    con.restraint = [2e-5, None, None]
    grad_fdiff = minimize(fun, x0=x0, method='SLSQP', options={'maxiter': 0}).jac
    grad = functools.partial(multipole_cost_function, structures=structures, constraints=con, calc_jac=True)(x0)[1]
    assert np.allclose(grad_fdiff, grad)


def test_run_alpha():
    this_file_location = os.path.dirname(os.path.abspath(__file__))

    constraintsfile = this_file_location + '/../../constraints/VAL_methyl_methyl.constraints.new'
    ref_files = [this_file_location + '/fchks_test/esp_VAL_methyl_methyl_0.h5' for i in range(6)]
    field_files = [
        this_file_location + '/fchks_test/esp_VAL_methyl_methyl_0_x+50.h5',
        this_file_location + '/fchks_test/esp_VAL_methyl_methyl_0_x-50.h5',
        this_file_location + '/fchks_test/esp_VAL_methyl_methyl_0_y+50.h5',
        this_file_location + '/fchks_test/esp_VAL_methyl_methyl_0_y-50.h5',
        this_file_location + '/fchks_test/esp_VAL_methyl_methyl_0_z+50.h5',
        this_file_location + '/fchks_test/esp_VAL_methyl_methyl_0_z-50.h5'
    ]

    #create constraints object
    con = constraints(constraintsfile)

    #load structure objects as defined from locationfile
    ref_structures = []
    field_structures = []

    for fname in ref_files:
        s = structure()
        s.load_h5(fname)
        ref_structures.append(s)

    for fname in field_files:
        s = structure()
        s.load_h5(fname)
        field_structures.append(s)

    x0 = con.get_polarizability_parameter_vector(isotropic=True)
    con.restraint = 1e-9

    fun = functools.partial(polarizability_cost_function,
                            structures=ref_structures,
                            fieldstructures=field_structures,
                            constraints=con)
    res = minimize(fun, x0=x0, method='SLSQP', tol=1e-12, options={'maxiter': 1000})
    alpha_check = np.array([
        8.50006728, 6.19302034, 8.73291484, 2.66001794, 8.13992296, 2.14752338, 8.81351293, 2.36159299, 8.14016149,
        8.15933284, 2.3477356, 8.09329647, 5.74683121, 2.40967811, 7.24353793, 1.50721465, 7.24980395, 1.70279374
    ])
    assert res.success
    np.allclose(res.x, alpha_check)
