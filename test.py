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
                if AAloaded[int(AAloaded[k,2])-1,0] == 'H' and AAloaded[k,0] == 'C':
                    print(AA[i], captype[j])
                assert AAloaded[int(AAloaded[k,2])-1,0] == AAloaded[k,0]


def test_parameterfile():
    # Test that atoms given the same atom type have identical parameters
    par = np.genfromtxt('FFparameterswithduplicates.csv', delimiter=',', dtype=str)
    checkdict = {}
    for i in range(1, len(par)):
        if par[i,0]+par[i,1]+'q' not in checkdict:
            checkdict[par[i,0]+par[i,1]+'q']   = float(par[i,2])
            #checkdict[par[i,0]+par[i,1]+'axx'] = float(par[i,3])
            #checkdict[par[i,0]+par[i,1]+'axy'] = float(par[i,4])
            #checkdict[par[i,0]+par[i,1]+'axz'] = float(par[i,5])
            #checkdict[par[i,0]+par[i,1]+'ayy'] = float(par[i,6])
            #checkdict[par[i,0]+par[i,1]+'ayz'] = float(par[i,7])
            #checkdict[par[i,0]+par[i,1]+'azz'] = float(par[i,8])
        else:
            assert checkdict[par[i,0]+par[i,1]+'q']   == float(par[i,2])
            #assert checkdict[par[i,0]+par[i,1]+'axx'] == float(par[i,3])
            #assert checkdict[par[i,0]+par[i,1]+'axy'] == float(par[i,4])
            #assert checkdict[par[i,0]+par[i,1]+'axz'] == float(par[i,5])
            #assert checkdict[par[i,0]+par[i,1]+'ayy'] == float(par[i,6])
            #assert checkdict[par[i,0]+par[i,1]+'ayz'] == float(par[i,7])
            #assert checkdict[par[i,0]+par[i,1]+'azz'] == float(par[i,8])


def test_parameterfile_totcharge():
    par = np.genfromtxt('FFparameterswithduplicates.csv', delimiter=',', dtype='U256')
    calcdict = {}
    checkdict =    {'NARG':2,
                    'nARG':1,
                    'CARG':0,
                    'cARG':1,
                    'ARG':1,
                    'NASN':1,
                    'nASN':0,
                    'CASN':-1,
                    'cASN':0,
                    'ASN':0,
                    'NASP':0,
                    'nASP':-1,
                    'CASP':-2,
                    'cASP':-1,
                    'ASP':-1,
                    'NCYS':1,
                    'nCYS':0,
                    'CCYS':-1,
                    'cCYS':0,
                    'CYS':0,
                    'NGLN':1,
                    'nGLN':0,
                    'CGLN':-1,
                    'cGLN':0,
                    'GLN':0,
                    'NGLU':0,
                    'nGLU':-1,
                    'CGLU':-2,
                    'cGLU':-1,
                    'GLU':-1,
                    'NHIS':2,
                    'nHIS':1,
                    'CHIS':0,
                    'cHIS':1,
                    'HIS':1,
                    'NILE':1,
                    'nILE':0,
                    'CILE':-1,
                    'cILE':0,
                    'ILE':0,
                    'NLEU':1,
                    'nLEU':0,
                    'CLEU':-1,
                    'cLEU':0,
                    'LEU':0,
                    'NMET':1,
                    'nMET':0,
                    'CMET':-1,
                    'cMET':0,
                    'MET':0,
                    'NPHE':1,
                    'nPHE':0,
                    'CPHE':-1,
                    'cPHE':0,
                    'PHE':0,
                    'NPRO':1,
                    'nPRO':0,
                    'CPRO':-1,
                    'cPRO':0,
                    'PRO':0,
                    'NSER':1,
                    'nSER':0,
                    'CSER':-1,
                    'cSER':0,
                    'SER':0,
                    'NTHR':1,
                    'nTHR':0,
                    'CTHR':-1,
                    'cTHR':0,
                    'THR':0,
                    'NTRP':1,
                    'nTRP':0,
                    'CTRP':-1,
                    'cTRP':0,
                    'TRP':0,
                    'NTYR':1,
                    'nTYR':0,
                    'CTYR':-1,
                    'cTYR':0,
                    'TYR':0,
                    'NVAL':1,
                    'nVAL':0,
                    'CVAL':-1,
                    'cVAL':0,
                    'VAL':0,
                    'NALA':1,
                    'nALA':0,
                    'CALA':-1,
                    'cALA':0,
                    'ALA':0,
                    'NASH':1,
                    'nASH':0,
                    'CASH':-1,
                    'cASH':0,
                    'ASH':0,
                    'NCYD':0,
                    'nCYD':-1,
                    'CCYD':-2,
                    'cCYD':-1,
                    'CYD':-1,
                    'NCYX':1,
                    'nCYX':0,
                    'CCYX':-1,
                    'cCYX':0,
                    'CYX':0,
                    'NGLH':1,
                    'nGLH':0,
                    'CGLH':-1,
                    'cGLH':0,
                    'GLH':0,
                    'NGLY':1,
                    'nGLY':0,
                    'CGLY':-1,
                    'cGLY':0,
                    'GLY':0,
                    'NHID':1,
                    'nHID':0,
                    'CHID':-1,
                    'cHID':0,
                    'HID':0,
                    'NHIE':1,
                    'nHIE':0,
                    'CHIE':-1,
                    'cHIE':0,
                    'HIE':0,
                    'NLYD':1,
                    'nLYD':0,
                    'CLYD':-1,
                    'cLYD':0,
                    'LYD':0,
                    'NLYS':2,
                    'nLYS':1,
                    'CLYS':0,
                    'cLYS':1,
                    'LYS':1}
                    
    for i in range(1, len(par)):
        if par[i,0] not in calcdict:
            calcdict[par[i,0]] = float(par[i,2])
        else:
            calcdict[par[i,0]] = float(par[i,2]) + calcdict[par[i,0]]
    for key in checkdict:
        print(key, calcdict[key])
        assert abs(checkdict[key] - calcdict[key]) < 10**-5
