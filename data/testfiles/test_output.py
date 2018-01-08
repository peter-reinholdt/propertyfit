import numpy as np
import json
import glob
import os

def test_constraint_files():
    # Test that atoms constrained to same charge are the same element
    this_file_location = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(this_file_location+'/../../constraints/*constraints')
    for file in files:
        with open(file, "r") as f:
            res = json.load(f)
        atomdict = {}
        for fragment in res["fragments"]:
            for i in range(0, len(fragment["atomnames"])):
                atomdict[fragment["atomindices"][i]] = fragment["atomnames"][i]
        for fragment in res["fragments"]:
            for symmetry in fragment["symmetries"]:
                atomcheck = atomdict[symmetry[0]]
                for i in symmetry:
                    assert atomcheck == atomdict[i]
                    
                    
def test_parameterfile():
    # Test that atoms given the same atom type have identical parameters
    this_file_location = os.path.dirname(os.path.abspath(__file__))
    par = np.genfromtxt(this_file_location+'/../FFparameterswithduplicates.csv', delimiter=',', dtype=str)
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


def test_parameterfile_totcharge():
    this_file_location = os.path.dirname(os.path.abspath(__file__))
    par = np.genfromtxt(this_file_location+'/../FFparameterswithduplicates.csv', delimiter=',', dtype='U256')
    calcdict = {}
    checkdict =    {'NARG':2,
                    'nARG':1,
                    'CARG':0,
                    'cARG':1,
                    'ARG':1,
                    'AARG':1,
                    'BARG':1,
                    'NASN':1,
                    'nASN':0,
                    'CASN':-1,
                    'cASN':0,
                    'ASN':0,
                    'AASN':0,
                    'BASN':0,
                    'NASP':0,
                    'nASP':-1,
                    'CASP':-2,
                    'cASP':-1,
                    'ASP':-1,
                    'AASP':-1,
                    'BASP':-1,
                    'NCYS':1,
                    'nCYS':0,
                    'CCYS':-1,
                    'cCYS':0,
                    'CYS':0,
                    'ACYS':0,
                    'BCYS':0,
                    'NGLN':1,
                    'nGLN':0,
                    'CGLN':-1,
                    'cGLN':0,
                    'GLN':0,
                    'AGLN':0,
                    'BGLN':0,
                    'NGLU':0,
                    'nGLU':-1,
                    'CGLU':-2,
                    'cGLU':-1,
                    'GLU':-1,
                    'AGLU':-1,
                    'BGLU':-1,
                    'NHIS':2,
                    'nHIS':1,
                    'CHIS':0,
                    'cHIS':1,
                    'HIS':1,
                    'AHIS':1,
                    'BHIS':1,
                    'NILE':1,
                    'nILE':0,
                    'CILE':-1,
                    'cILE':0,
                    'ILE':0,
                    'AILE':0,
                    'BILE':0,
                    'NLEU':1,
                    'nLEU':0,
                    'CLEU':-1,
                    'cLEU':0,
                    'LEU':0,
                    'ALEU':0,
                    'BLEU':0,
                    'NMET':1,
                    'nMET':0,
                    'CMET':-1,
                    'cMET':0,
                    'MET':0,
                    'AMET':0,
                    'BMET':0,
                    'NPHE':1,
                    'nPHE':0,
                    'CPHE':-1,
                    'cPHE':0,
                    'PHE':0,
                    'APHE':0,
                    'BPHE':0,
                    'NPRO':1,
                    'nPRO':0,
                    'CPRO':-1,
                    'cPRO':0,
                    'PRO':0,
                    'APRO':0,
                    'BPRO':0,
                    'NSER':1,
                    'nSER':0,
                    'CSER':-1,
                    'cSER':0,
                    'SER':0,
                    'ASER':0,
                    'BSER':0,
                    'NTHR':1,
                    'nTHR':0,
                    'CTHR':-1,
                    'cTHR':0,
                    'THR':0,
                    'ATHR':0,
                    'BTHR':0,
                    'NTRP':1,
                    'nTRP':0,
                    'CTRP':-1,
                    'cTRP':0,
                    'TRP':0,
                    'ATRP':0,
                    'BTRP':0,
                    'NTYR':1,
                    'nTYR':0,
                    'CTYR':-1,
                    'cTYR':0,
                    'TYR':0,
                    'ATYR':0,
                    'BTYR':0,
                    'NVAL':1,
                    'nVAL':0,
                    'CVAL':-1,
                    'cVAL':0,
                    'VAL':0,
                    'AVAL':0,
                    'BVAL':0,
                    'NALA':1,
                    'nALA':0,
                    'CALA':-1,
                    'cALA':0,
                    'ALA':0,
                    'AALA':0,
                    'BALA':0,
                    'NASH':1,
                    'nASH':0,
                    'CASH':-1,
                    'cASH':0,
                    'ASH':0,
                    'AASH':0,
                    'BASH':0,
                    'NCYD':0,
                    'nCYD':-1,
                    'CCYD':-2,
                    'cCYD':-1,
                    'CYD':-1,
                    'ACYD':-1,
                    'BCYD':-1,
                    'NCYX':1,
                    'nCYX':0,
                    'CCYX':-1,
                    'cCYX':0,
                    'CYX':0,
                    'ACYX':0,
                    'BCYX':0,
                    'NGLH':1,
                    'nGLH':0,
                    'CGLH':-1,
                    'cGLH':0,
                    'GLH':0,
                    'AGLH':0,
                    'BGLH':0,
                    'NGLY':1,
                    'nGLY':0,
                    'CGLY':-1,
                    'cGLY':0,
                    'GLY':0,
                    'AGLY':0,
                    'BGLY':0,
                    'NHID':1,
                    'nHID':0,
                    'CHID':-1,
                    'cHID':0,
                    'HID':0,
                    'AHID':0,
                    'BHID':0,
                    'NHIE':1,
                    'nHIE':0,
                    'CHIE':-1,
                    'cHIE':0,
                    'HIE':0,
                    'AHIE':0,
                    'BHIE':0,
                    'NLYD':1,
                    'nLYD':0,
                    'CLYD':-1,
                    'cLYD':0,
                    'LYD':0,
                    'ALYD':0,
                    'BLYD':0,
                    'NLYS':2,
                    'nLYS':1,
                    'CLYS':0,
                    'cLYS':1,
                    'LYS':1,
                    'ALYS':1,
                    'BLYS':1}
                    
    for i in range(1, len(par)):
        if par[i,0] not in calcdict:
            calcdict[par[i,0]] = float(par[i,2])
        else:
            calcdict[par[i,0]] = float(par[i,2]) + calcdict[par[i,0]]
    for key in checkdict:
        assert abs(checkdict[key] - calcdict[key]) < 10**-8
