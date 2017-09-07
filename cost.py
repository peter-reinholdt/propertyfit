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
    # are forced in such away that nummerical problems can arise
    # cost function might be discontineous?
    # Atoms to be constrained to the same charge, can be found
    # in constraints file;
    # Constraint file = [Atom, index, same charge index, start charge guess]
    #
    # q    = input charge
    # qout = charge used in RMSD calculation
    # capq = Change in cap atoms to get total capping charge to zero
    #
    # checkCYX, is used to see if AA is CYX. For CYX, checkCYX=1
    # cehckTERMINAL, is used to determine terminal type.
    #     1 = methylcharged
    #     2 = methylneutral
    #     3 = chargedmethyl
    #     4 = neutralmethyl
    #
    # #####################################################
    
    # Set cappings:
    if checkTERMINAL == 0:
        cap1 = 6
        cap2 = 6
    elif checkTERMINAL == 1 or checkTERMINAL == 2:
        cap1 = 0
        cap2 = 6
    elif checkTERMINAL == 3 or checkTERMINAL == 4:
        cap1 = 6
        cap2 = 0
    if checkCYX == 1:
        cap2 += 5

    # Make same atoms same charge
    qout=q
    for i in range(0, len(constraints)):
        qout[i] = qout[int(constraints[i,2])-1]

    # Assign constraints
    # Total charge on caps is set to zero
    capq = 0
    if cap1 != 0:
        capq += qout[0:cap1].sum()
    if cap2 != 0:
        capq += qout[-cap2:].sum()
    if cap1 != 0:
        qout[0:cap1] -= (capq)/(cap1+cap2)
    if cap2 != 0:
        qout[-cap2:] -= (capq)/(cap1+cap2)
    
    # Total molecular charge set to qtot
    atoms = len(qout)
    qout[cap1:atoms-cap2] -= (qout[cap1:atoms-cap2].sum()-qtot)/(atoms-cap1-cap2)

    return np.sqrt(np.average([chargefit.esp_sum_squared_error(s.rinvmat, s.esp_grid_qm, qout) for s in structures]))

 

def fit(AA):
    global constraints
    global structures
    # Ugly way to load in constraints, but it works for now
    if checkTERMINAL == 1:
        constraints = np.genfromtxt('/work/sdujk/kjellgren/propertyfit/constraints/'+AA+'methylchargedidx.csv', delimiter=';', dtype=str)
    elif checkTERMINAL == 2:
        constraints = np.genfromtxt('/work/sdujk/kjellgren/propertyfit/constraints/'+AA+'methylneutralidx.csv', delimiter=';', dtype=str)
    elif checkTERMINAL == 3:
        constraints = np.genfromtxt('/work/sdujk/kjellgren/propertyfit/constraints/'+AA+'chargedmethylidx.csv', delimiter=';', dtype=str)
    elif checkTERMINAL == 4:
        constraints = np.genfromtxt('/work/sdujk/kjellgren/propertyfit/constraints/'+AA+'neutralmethylidx.csv', delimiter=';', dtype=str)
    else:
        constraints = np.genfromtxt('/work/sdujk/kjellgren/propertyfit/constraints/'+AA+'idx.csv', delimiter=';', dtype=str)
    for i in range(0, len(constraints)):
            constraints[i,1] = int(constraints[i,1])
            constraints[i,2] = int(constraints[i,2])
            constraints[i,3] = float(constraints[i,3])
    
    # Get initial charge guess
    q0 = get_q0(constraints)
    
    # Get structures from fchk files
    # Change below to something more meaningfull
    if checkTERMINAL == 1:
        frsx = glob.glob("/work/sdujk/reinholdt/IAKE805/charge/datacaps/"+str(AA)+"_methyl_charged/*.fchk.s")
    elif checkTERMINAL == 2:
        frsx = glob.glob("/work/sdujk/reinholdt/IAKE805/charge/datacaps/"+str(AA)+"_methyl_neutral/*.fchk.s")
    elif checkTERMINAL == 3:
        frsx = glob.glob("/work/sdujk/reinholdt/IAKE805/charge/datacaps/"+str(AA)+"_charged_methyl/*.fchk.s")
    elif checkTERMINAL == 4:
        frsx = glob.glob("/work/sdujk/reinholdt/IAKE805/charge/datacaps/"+str(AA)+"_neutral_methyl/*.fchk.s")
    else:
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
        if checkTERMINAL == 1:
            file = open(str(AA)+'methylcharged_q_out.txt','w')
        elif checkTERMINAL == 2:
            file = open(str(AA)+'methylneutral_q_out.txt','w')
        elif checkTERMINAL == 3:
            file = open(str(AA)+'chargedmethyl_q_out.txt','w')
        elif checkTERMINAL == 4:
            file = open(str(AA)+'neutralmethyl_q_out.txt','w')
        else:
            file = open(str(AA)+'_q_out.txt','w')
        res = scipy.optimize.minimize(cost, q0, method='SLSQP', options={'ftol':1e-10})
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


def run(AA):
    global qtot
    qtot = 0
    if AA == 'LYS' or AA == 'ARG' or AA == 'HIS':
        qtot += 1
    elif AA == 'ASP' or AA == 'GLU' or AA =='CYD':
        qtot += -1
    if checkTERMINAL == 1:
        qtot += 1
    elif checkTERMINAL == 3:
        qtot += -1
    fit(AA)

#AAlist = ['ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'ALA', 'ASH', 'CYD', 'CYX', 'GLH', 'GLY', 'HID', 'HIE', 'LYD', 'LYS']
#terminals = ['methylcharged','methylneutral','chargedmethyl','neutralmethyl', 'None']
AAlist = ['PLACEHOLDER']
terminals = ['methylcharged','methylneutral','chargedmethyl','neutralmethyl', 'None']

AAlist[0] = str(sys.argv[1])

global checkCYX
global checkTERMINAL
for i in range(len(AAlist)):
    if AAlist[i] == 'CYX':
        checkCYX = 1
    else:
        checkCYX = 0
    for j in range(len(terminals)):
        if terminals[j] == 'methylcharged':
            checkTERMINAL = 1
        elif terminals[j] == 'methylneutral':
            checkTERMINAL = 2
        elif terminals[j] == 'chargedmethyl':
            checkTERMINAL = 3
        elif terminals[j] == 'neutralmethyl':
            checkTERMINAL = 4
        else:
            checkTERMINAL = 0
        run(AAlist[i])
