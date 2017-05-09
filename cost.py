import chargefit
import numpy as np
import scipy.optimize

structures = chargefit.loadfchks("test/VAL_[^xyz]*.fchk")
[s.compute_grid() for s in structures]
[s.compute_rinvmat() for s in structures]
[s.compute_qm_esp() for s in structures]

def get_q0():
    # #####################################################
    #
    # Gets the start guess of the charges, fourth coloum
    # in the constraints files 
    #
    # #####################################################
    return constraints[:,3]

def cost(q):
    # #####################################################
    #
    # Put hard constraints on the total charge, and atoms
    # that is chosen to have the same charge. The constraints
    # is forced in such away that nummerical problems can arise
    # cost function might be discontineous?
    # Atoms to be constrained to the same charge, can be found
    # in constraints file;
    # Constraint file = [Atom, index, same charge index, start charge guess]
    #
    # q    = input charge
    # qout = charge used in RMSD calculation
    # capq = Change in cap atoms to get total capping charge to zero
    #
    # #####################################################
    
    # Make same atoms same charge
    qout=q
    for i in range(0, len(constraints)):
        qout[i] = qout[int(constraints[i,2])-1]

    # Assign constraints
    # Total charge on caps is set to zero
    capq = qout[0:6].sum()+qout[-6:].sum()
    qout[0:6] -= (capq)/12
    qout[-6:] -= (capq)/12
    
    # Total molecular charge set to qtot
    atoms = len(qout)
    qout[6:atoms-6] -= (qout[6:atoms-6].sum()-qtot)/(atoms-12)

    return np.average([chargefit.esp_sum_squared_error(s.rinvmat, s.esp_grid_qm, qout) for s in structures])

def return_q(q):
    # #####################################################
    #
    # Charges are changed inside cost function after optimization
    # call, thus the charges need to be changed in the same way
    # one last time to get the proper charges.
    #
    # #####################################################
    # Make same atoms same charge
    qout=q
    for i in range(0, len(constraints)):
        qout[i] = qout[int(constraints[i,2])-1]

    # Assign constraints
    # Total charge on caps is set to zero
    capq = qout[0:6].sum()+qout[-6:].sum()
    qout[0:6] -= (capq)/12
    qout[-6:] -= (capq)/12
    
    # Total molecular charge set to qtot
    atoms = len(qout)
    qout[6:atoms-6] -= (qout[6:atoms-6].sum()-qtot)/(atoms-12)
    
    return qout


def fit_induced_potential():
    pass
    #stuff

# AA = amino acid, give name as string // make it as input
AA = 'GLY'
# qtot = total charge of the amino acid, give number // make it as input
qtot = 0.0
constraints = np.genfromtxt('constraints/'+AA+'idx.csv', delimiter=';')
q0 = get_q0()
res = scipy.optimize.minimize(cost, q0, method='SLSQP')
print(res)
print(return_q(res['x']))

s.write_xyz('mol.xyz')
s.write_grid('grid.xyz')
