import chargefit
import numpy as np
import scipy.optimize
import sys
import glob
from conversions import name2number
import re

def get_q0(constraints):
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
    
    
def costCYX(q):
    # #####################################################
    #
    # Special case for CYX, capping on sulfur
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
    capq = qout[0:6].sum()+qout[-11:].sum()
    qout[0:6] -= (capq)/17
    qout[-11:] -= (capq)/17
    
    # Total molecular charge set to qtot
    atoms = len(qout)
    qout[6:atoms-11] -= (qout[6:atoms-11].sum()-qtot)/(atoms-17)

    return np.average([chargefit.esp_sum_squared_error(s.rinvmat, s.esp_grid_qm, qout) for s in structures])    


# AA = amino acid, give name as string
#AA = str(sys.argv[1])
# qtot = total charge of the amino acid, give number
#qtot = float(sys.argv[2])
def fit(AA):
    global constraints
    global structures
    # Ugly way to load in constraints, but it works for now
    constraints = np.genfromtxt('/work/sdujk/kjellgren/propertyfit/constraints/'+AA+'idx.csv', delimiter=';', dtype=str)
    for i in range(0, len(constraints)):
            constraints[i,1] = int(constraints[i,1])
            constraints[i,2] = int(constraints[i,2])
            constraints[i,3] = float(constraints[i,3])
    
    # Get initial charge guess
    q0 = get_q0(constraints)
    
    # Get structures from fchk files
    # Change below to something more meaningfull
    frsx = glob.glob("/work/sdujk/reinholdt/IAKE805/charge/data/"+str(AA)+"/*.fchk.s")
    frs = []
    for i in frsx:
        x = re.split("/|\.", i)[-3]
        frs.append(i)
    structures = [chargefit.load_file(f) for f in frs]

    #Check if sys arg and fchk match
    check = 0
    for i in structures:
        for j in range(0, len(i.numbers)):
            if i.numbers[j] != name2number[constraints[j,0]]:
                check = 1
                print('FATAL ERROR: Atom number '+str(j+1)+' does not match')
                break
    
    if check == 0:
        file = open(str(AA)+'_q_out.txt','w')
        if AA == 'CYX':
            res = scipy.optimize.minimize(costCYX, q0, method='SLSQP')
        else:
            res = scipy.optimize.minimize(cost, q0, method='SLSQP')
        for key in res:
            file.write(str(key)+'    '+str(res[key]))
            file.write('\n')
        file.write('\n')
        for i in range(0, len(res['x'])):
             file.write(str(res['x'][i]))
             file.write('\n')
        file.close()
    else:
        if check == 1:
            print('FATAL ERROR: Atom(s) in given molecule, does not match atom(s) in fchk-file(s)')
        else:
            print('FATAL ERROR: UNKNOWN')


#AAlist = ['ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'ALA', 'ASH', 'CYD', 'CYX', 'GLH', 'GLY', 'HID', 'HIE', 'LYD', 'LYS']
def run(AAlist):
    global qtot
    if AAlist == 'LYS' or AAlist == 'ARG' or AAlist == 'HIS':
        qtot = 1
    elif AAlist == 'ASP' or AAlist == 'GLU' or AAlist =='CYD':
        qtot = -1
    else:
        qtot = 0
    fit(AAlist)

AA = str(sys.argv[1])
run(AA)

