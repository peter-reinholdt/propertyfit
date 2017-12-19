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

    h = hashlib.sha1()
    with open(this_file_location+'/../../run_charge.py','rb') as file:
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h.update(chunk)
    hashout = h.hexdigest()
    
    # Checks that run_charge have not been altered, and should therefore 
    #  give the same result as this test.
    assert hashout == '1f1fbd0f1a8d99898c134307193bf40d28619d1f'
    
    
    constraintsfile = this_file_location+'/../../constraints/VAL.constraints'
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
    
    con.restraint = 1.0
    q0 = con.q0

    fun = functools.partial(charge_cost_function, structures=structures, constraints=con)
    res = minimize(fun, x0=q0, method='SLSQP')
    q_check = [0.502411534, -0.460790832, -0.469714568, -0.288719066, 0.174618232, -0.083053924, 0.126948577, 0.029186223, 0.501271583, 0.086886897, -0.446373001, -0.477342324, -0.335214414, 0.250139278, -0.332115715]
    for i in range(0, len(res.x)):
        assert abs(res.x[i] - q_check[i]) < 10**-8


def test_run_alpha():
    this_file_location = os.path.dirname(os.path.abspath(__file__))

    h = hashlib.sha1()
    with open(this_file_location+'/../../run_alpha.py','rb') as file:
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h.update(chunk)
    hashout = h.hexdigest()
    
    # Checks that run_charge have not been altered, and should therefore 
    #  give the same result as this test.
    assert hashout == '57e035989a425f080cb14f41191746dc19e10dfb'
    
    constraintsfile = this_file_location+'/../../constraints/VAL.constraints'
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
        assert abs(res.x[i] - alpha_check[i]) < 10**-8