import glob
import os
import hashlib
import sys
import numpy as np
import functools
from scipy.optimize import minimize

this_file_location = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,this_file_location+'/../../')
from propertyfit.structures import structure, constraints
from propertyfit.costfunctions import charge_cost_function, isopol_cost_function

def test_run_charge():
    this_file_location = os.path.dirname(os.path.abspath(__file__))    
    
    constraintsfile = this_file_location+'/../../constraints/VAL_methyl_methyl.constraints'
    files = [this_file_location+'/fchks_test/VAL_0.fchk.h5']
    
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
    
    con.restraint = 2.0e-7
    q0 = con.q0

    fun = functools.partial(charge_cost_function, structures=structures, constraints=con)
    res = minimize(fun, x0=q0, method='SLSQP', tol=1e-17, options={'maxiter': 1000})
    q_check = np.array([ 0.50250079, -0.4609129 , -0.46979605, -0.28898077,  0.17460721,
       -0.08110295,  0.1266726 ,  0.02618351,  0.5006874 ,  0.08786678,
       -0.44565655, -0.4772865 , -0.33626983,  0.24962138, -0.32482822])
    assert res.success
    for i in range(0, len(res.x)):
        assert abs(res.x[i] - q_check[i]) < 10**-3


def test_run_alpha():
    this_file_location = os.path.dirname(os.path.abspath(__file__))
    
    constraintsfile = this_file_location+'/../../constraints/VAL_methyl_methyl.constraints'
    files = np.array([[this_file_location+'/fchks_test/VAL_0.fchk.h5', 
            this_file_location+'/fchks_test/VAL_0_x+50.fchk.h5',
            this_file_location+'/fchks_test/VAL_0_x-50.fchk.h5',
            this_file_location+'/fchks_test/VAL_0_y+50.fchk.h5',
            this_file_location+'/fchks_test/VAL_0_y-50.fchk.h5',
            this_file_location+'/fchks_test/VAL_0_z+50.fchk.h5',
            this_file_location+'/fchks_test/VAL_0_z-50.fchk.h5']])
    ref_files = files[:,0]
    field_files = files[:,1]
    
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
    
    
    #use partial to wrap cost function, so we only need a single atest argument (and not constraints, structures)
    #then we can call fun(atest) instead of isopol_cost_function(atest, structures, fieldstructures, constraints)
    
    #read initial parameters from q0
    a0 = np.zeros(con.nparametersa)
    
    fun = functools.partial(isopol_cost_function, structures=ref_structures, fieldstructures=field_structures, constraints=con)
    res = minimize(fun, x0=a0, method='SLSQP')
    alpha_check = [8.274135822, 11.74197291, 2.809621816, 3.961664646, 7.850222049, 9.080497659, 5.428903941, 5.366737385, 4.354975905, 4.867776458, 8.293173471, 3.11332895, 6.144178157, 4.038802156, 1.442260717, -0.558596838, 1.294815758, 4.014986792]
    
    for i in range(0, len(res.x)):
        assert abs(res.x[i] - alpha_check[i]) < 10**-3
