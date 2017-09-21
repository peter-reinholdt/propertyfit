def potentialPBD(V, molecule, filename):
    # #####################################################
    #
    # Function to write ESP and error in ESP to a pdb file,
    # that can be visualized in VMD.
    #
    # V is the points where to the ESP have been calculated.
    # V is saved in the pdb as the atom He
    # V[0,:] = x
    # V[1,:] = y
    # V[2,:] = z
    # V[3,:] = ESP
    # V[4,:] = ESP error
    #
    # molcule is the coordinates of the molecule
    # it is assumed that it has no header
    # molcule[0,:] = atom name
    # molcule[1,:] = x
    # molcule[1,:] = y
    # molcule[1,:] = z
    #
    # filename is the name of the pdb file, such that:
    # file = "filename"_potential.pdb
    #
    # #####################################################
    
    #AtomNrtoName =  {1: 'H',2: 'He',3: 'Li',4: 'Be',5: 'B',6: 'C',7: 'N',8: 'O',9: 'F',10: 'Ne',11: 'Na',12: 'Mg',13: 'Al',14: 'Si',15: 'P',16: 'S',17: 'Cl',18: 'Ar',19: 'K',20: 'Ca',21: 'Sc',22: 'Ti',23: 'V',24: 'Cr',25: 'Mn',26: 'Fe',27: 'Co',28: 'Ni',29: 'Cu',30: 'Zn',31: 'Ga',32: 'Ge',33: 'As',34: 'Se',35: 'Br',36: 'Kr',37: 'Rb',38: 'Sr',39: 'Y',40: 'Zr',41: 'Nb',42: 'Mo',43: 'Tc',44: 'Ru',45: 'Rh',46: 'Pd',47: 'Ag',48: 'Cd',49: 'In',50: 'Sn',51: 'Sb',52: 'Te',53: 'I',54: 'Xe',55: 'Cs',56: 'Ba',57: 'La',58: 'Ce',59: 'Pr',60: 'Nd',61: 'Pm',62: 'Sm',63: 'Eu',64: 'Gd',65: 'Tb',66: 'Dy',67: 'Ho',68: 'Er',69: 'Tm',70: 'Yb',71: 'Lu',72: 'Hf',73: 'Ta',74: 'W',75: 'Re',76: 'Os',77: 'Ir',78: 'Pt',79: 'Au',80: 'Hg',81: 'Tl',82: 'Pb',83: 'Bi',84: 'Po',85: 'At',86: 'Rn',87: 'Fr',88: 'Ra',89: 'Ac',90: 'Th',91: 'Pa',92: 'U',93: 'Np',94: 'Pu',95: 'Am',96: 'Cm',97: 'Bk',98: 'Cf',99: 'Es',100: 'Fm',101: 'Md',102: 'No',103: 'Lr',104: 'Rf',105: 'Db',106: 'Sg',107: 'Bh',108: 'Hs',109: 'Mt'}
    
    # scaling of ESP and ESP error to fit within the PDB format
    scaleESP = 100
    scaleESPerror = 10000
    
    #Write potential to pdb
    f = open(filename+'_potential.pdb', 'w+')
    idx = 1
    for i in range(0, len(molecule)):
        f.write('{:>6}'.format('HETATM'))
        f.write('{:>5}'.format(str(idx)))
        f.write('{:>1}'.format(' '))
        #atom = AtomNrtoName[molecule[i,0]]
        atom = molecule[i,0]
        f.write('{:>4}'.format(atom))
        f.write('{:>1}'.format(' '))
        f.write('{:>3}'.format('MOL'))
        f.write('{:>1}'.format(' '))
        f.write('{:>1}'.format(' '))
        f.write('{:>4}'.format(' '))
        f.write('{:>1}'.format(' '))
        f.write('{:>3}'.format(' '))
        f.write('{: 8.3f}'.format(molecule[i,1]))
        f.write('{: 8.3f}'.format(molecule[i,2]))
        f.write('{: 8.3f}'.format(molecule[i,3]))
        f.write('{: 6.2f}'.format(1))
        f.write('{: 6.2f}'.format(1))
        f.write('{:>6}'.format(' '))
        f.write('{:>4}'.format(' '))
        f.write('{:>2}'.format(' '))
        f.write('{:>2}'.format(' '))
        f.write('\n')
        idx += 1
    for i in range(0, len(V)):
        f.write('{:>6}'.format('HETATM')) #Record name     "HETATM" 
        f.write('{:>5}'.format(str(idx))) #Integer         Atom serial number. 
        f.write('{:>1}'.format(' '))      #Blanck space
        f.write('{:>4}'.format('He'))     #Atom            Atom name    
        f.write('{:>1}'.format(' '))      #Character       Alternate location indicator 
        f.write('{:>3}'.format('POT'))    #Residue name    Residue name 
        f.write('{:>1}'.format(' '))      #Blanck space
        f.write('{:>1}'.format(' '))      #Character       Chain identifier 
        f.write('{:>4}'.format(' '))      #Integer         Residue sequence number   
        f.write('{:>1}'.format(' '))      #AChar           Code for insertion of residues 
        f.write('{:>3}'.format(' '))      #Blank space
        f.write('{: 8.3f}'.format(V[i,0]))#Real(8.3)       Orthogonal coordinates for X    
        f.write('{: 8.3f}'.format(V[i,1]))#Real(8.3)       Orthogonal coordinates for Y   
        f.write('{: 8.3f}'.format(V[i,2]))#Real(8.3)       Orthogonal coordinates for Z   
        f.write('{: 6.2f}'.format(V[i,3]*scaleESP))#Real(6.2)       Occupancy 
        f.write('{: 6.2f}'.format(V[i,4]*scaleESPerror))#Real(6.2)  Temperature factor 
        f.write('{:>6}'.format(' ')) #Blank space
        f.write('{:>4}'.format(' ')) #LString(4)      Segment identifier, left-justified  
        f.write('{:>2}'.format(' ')) #LString(2)      Element symbol, right-justified 
        f.write('{:>2}'.format(' ')) #LString(2)      Charge on the atom 
        f.write('\n')
        idx += 1
    f.close()