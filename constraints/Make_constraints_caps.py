import numpy as np

def runConstraints(AA):
    # ########################################
    #
    # Makes constraint indexing for charge
    # and polarizability fit, for terminal
    # amino acids.
    # AA = amino acid three letter abrivation
    #
    # ########################################
    for i in range(1, 5):
        # Four cases:
        # I  : methyl charged
        # II : methyl neutral
        # III: charged methyl
        # IV : neutral methyl
        AAidx = np.genfromtxt(str(AA)+'idx.csv',delimiter=';',dtype=str)    
        if i == 1:
            for j in range(len(AAidx)):
                AAidx[j,1] = str(int(AAidx[j,1])-4)
                AAidx[j,2] = str(int(AAidx[j,2])-4)
            
            AAidx = np.delete(AAidx,[0,1,2,3],0)
            AAidx[0,2] = AAidx[1,2] = '1'
            output = open(str(AA)+'methylcharged.csv', 'w')
            for j in range(len(AAidx)):
                output.write(AAidx[j,0]+';'+AAidx[j,1]+';'+AAidx[j,2]+';'+AAidx[j,3]+'\n')
                
        
        elif i == 2:
            for j in range(len(AAidx)):
                AAidx[j,1] = str(int(AAidx[j,1])-5)
                AAidx[j,2] = str(int(AAidx[j,2])-5)
            
            AAidx = np.delete(AAidx,[0,1,2,3,4],0)
            AAidx[0,2] = '1'
            output = open(str(AA)+'methylneutral.csv', 'w')
            for j in range(len(AAidx)):
                output.write(AAidx[j,0]+';'+AAidx[j,1]+';'+AAidx[j,2]+';'+AAidx[j,3]+'\n')
            
        elif i == 3:
            AAidx = np.delete(AAidx,[len(AAidx)-1,len(AAidx)-2,len(AAidx)-3,len(AAidx)-4,len(AAidx)-5],0)
            AAidx[-1,0] = 'O'
            AAidx[-1,1] = AAidx[-1,2] = str(int(AAidx[-2,1])+1)
            output = open(str(AA)+'chargedmethyl.csv', 'w')
            for j in range(len(AAidx)):
                output.write(AAidx[j,0]+';'+AAidx[j,1]+';'+AAidx[j,2]+';'+AAidx[j,3]+'\n')
        
        else:
            AAidx = np.delete(AAidx,[len(AAidx)-1,len(AAidx)-2,len(AAidx)-3,len(AAidx)-4],0)
            AAidx[-2,0] = 'O'
            AAidx[-2,1] = AAidx[-2,2] = str(int(AAidx[-3,1])+1)
            AAidx[-1,0] = 'H'
            AAidx[-1,1] = AAidx[-1,2] = str(int(AAidx[-3,1])+2)
            output = open(str(AA)+'neutralmethyl.csv', 'w')
            for j in range(len(AAidx)):
                output.write(AAidx[j,0]+';'+AAidx[j,1]+';'+AAidx[j,2]+';'+AAidx[j,3]+'\n')
        
AAlist = ['ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'ALA', 'ASH', 'CYD', 'CYX', 'GLH', 'GLY', 'HID', 'HIE', 'LYD', 'LYS']
for k in range(len(AAlist)):
    runConstraints(AAlist[k])