import numpy as np

def test_constraints():
    AA = ['ALA', 'ARG', 'ASH', 'ASN', 'ASP',  'CYD', 'CYS', 'CYX', 'GLH', 'GLN', 'GLU', 'GLY', 'HID', 'HIE', 'HIS', 'ILE', 'LEU', 'LYD', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    for i in range(0, len(AA)):
        AAloaded = np.genfromtxt('constraints/'+str(AA[i])+'idx.csv', delimiter=';', dtype='str')
        for j in range(0, len(AAloaded)):
            assert AAloaded[int(AAloaded[i,2])-1,0] == AAloaded[i,0]

