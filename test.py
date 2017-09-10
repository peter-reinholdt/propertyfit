import numpy as np


def test_constraints():
    # Test that atoms constrained to same charge are the same element
    AA = ['ALA', 'ARG', 'ASH', 'ASN', 'ASP',  'CYD', 'CYS', 'CYX', 'GLH', 'GLN', 'GLU', 'GLY', 'HID', 'HIE', 'HIS', 'ILE', 'LEU', 'LYD', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    for i in range(0, len(AA)):
        AAloaded = np.genfromtxt('constraints/'+str(AA[i])+'idx.csv', delimiter=';', dtype='str')
        for j in range(0, len(AAloaded)):
            assert AAloaded[int(AAloaded[i,2])-1,0] == AAloaded[i,0]


def test_constraints_caps():
    # Test that atoms constrained to same charge are the same element
    AA = ['ALA', 'ARG', 'ASH', 'ASN', 'ASP',  'CYD', 'CYS', 'CYX', 'GLH', 'GLN', 'GLU', 'GLY', 'HID', 'HIE', 'HIS', 'ILE', 'LEU', 'LYD', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    captype = ['methylcharged','methylneutral','chargedmethyl','neutralmethyl']
    for i in range(0, len(AA)):
        for j in range(0, len(captype)):
            AAloaded = np.genfromtxt('constraints/'+str(AA[i])+str(captype[j])+'idx.csv', delimiter=';', dtype='str')
            for k in range(0, len(AAloaded)):
                assert AAloaded[int(AAloaded[k,2])-1,0] == AAloaded[k,0]


def test_parameterfile():
    # Test that atoms given the same atom type have identical parameters
    par = np.genfromtxt('FFparameters.csv', delimiter=';', dtype=str)
    checkdict = {}
    for i in range(1, len(par)):
        if par[i,0]+par[i,1]+'q' not in checkdict:
            checkdict[par[i,0]+par[i,1]+'q']   = float(par[i,2])
            checkdict[par[i,0]+par[i,1]+'axx'] = float(par[i,3])
            checkdict[par[i,0]+par[i,1]+'axy'] = float(par[i,4])
            checkdict[par[i,0]+par[i,1]+'axz'] = float(par[i,5])
            checkdict[par[i,0]+par[i,1]+'ayy'] = float(par[i,6])
            checkdict[par[i,0]+par[i,1]+'ayz'] = float(par[i,7])
            checkdict[par[i,0]+par[i,1]+'azz'] = float(par[i,8])
        else:
            assert checkdict[par[i,0]+par[i,1]+'q']   == float(par[i,2])
            assert checkdict[par[i,0]+par[i,1]+'axx'] == float(par[i,3])
            assert checkdict[par[i,0]+par[i,1]+'axy'] == float(par[i,4])
            assert checkdict[par[i,0]+par[i,1]+'axz'] == float(par[i,5])
            assert checkdict[par[i,0]+par[i,1]+'ayy'] == float(par[i,6])
            assert checkdict[par[i,0]+par[i,1]+'ayz'] == float(par[i,7])
            assert checkdict[par[i,0]+par[i,1]+'azz'] == float(par[i,8])