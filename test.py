import numpy as np

def test_constraints():
    AA = ['ALA', 'ARG', 'ASH', 'ASN', 'ASP',  'CYD', 'CYS', 'CYX', 'GLH', 'GLN', 'GLU', 'GLY', 'HID', 'HIE', 'HIS', 'ILE', 'LEU', 'LYD', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    for i in range(0, len(AA)):
        AAloaded = np.genfromtxt('constraints/'+str(AA[i])+'idx.csv', delimiter=';', dtype='str')
        for j in range(0, len(AAloaded)):
            assert AAloaded[int(AAloaded[i,2])-1,0] == AAloaded[i,0]

def test_constraints_caps():
    AA = ['ALA', 'ARG', 'ASH', 'ASN', 'ASP',  'CYD', 'CYS', 'CYX', 'GLH', 'GLN', 'GLU', 'GLY', 'HID', 'HIE', 'HIS', 'ILE', 'LEU', 'LYD', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    captype = ['methylcharged','methylneutral','chargedmethyl','neutralmethyl']
    for i in range(0, len(AA)):
        for j in range(0, len(captype)):
            AAloaded = np.genfromtxt('constraints/'+str(AA[i])+str(captype[j])+'idx.csv', delimiter=';', dtype='str')
            for k in range(0, len(AAloaded)):
                assert AAloaded[int(AAloaded[k,2])-1,0] == AAloaded[k,0]
