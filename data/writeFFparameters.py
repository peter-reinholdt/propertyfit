import numpy as np


def makeparameterfiler():
    parameters = np.genfromtxt('FFparameterstemplate.csv',delimiter=';',dtype='U256')
    current = 'None'
    for i in range(1,len(parameters)):
        
        if current != parameters[i,0]:

            if parameters[i,0][0] == 'C' and parameters[i,0][1] != 'Y':
                with open('fittedparameters/charges_'+parameters[i,0][1:]+'_charged_methyl.out') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'Fin':
                        charges = fitoutput[j+1+6:len(fitoutput)]
                        for k in range(0, len(charges)):
                            charges[k] = charges[k][:-2]
                        alpha  = np.genfromtxt('fittedparameters/alpha_'+parameters[i,0][1:]+'_charged_methyl.out.txt')[6:]
                        if parameters[i,0][1:] == 'CYX':
                            charges = np.delete(charges, [10, 11, 12, 13, 14])
                            alpha = np.delete(alpha, [10, 11, 12, 13, 14])
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,4] = 0.0
                            parameters[i+k,5] = 0.0
                            parameters[i+k,6] = alpha[k]
                            parameters[i+k,7] = 0.0
                            parameters[i+k,8] = alpha[k]
                                
            elif parameters[i,0][0] == 'c':
                with open('fittedparameters/restraints_'+parameters[i,0][1:]+'_neutral_methyl.out') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'Fin':
                        charges = fitoutput[j+1+6:len(fitoutput)]
                        for k in range(0, len(charges)):
                            charges[k] = charges[k][:-2]
                        alpha  = np.genfromtxt('fittedparameters/alpha_'+parameters[i,0][1:]+'_neutral_methyl.out.txt')[6:]
                        if parameters[i,0][1:] == 'CYX':
                            charges = np.delete(charges, [10, 11, 12, 13, 14])
                            alpha = np.delete(alpha, [10, 11, 12, 13, 14])
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,4] = 0.0
                            parameters[i+k,5] = 0.0
                            parameters[i+k,6] = alpha[k]
                            parameters[i+k,7] = 0.0
                            parameters[i+k,8] = alpha[k]
                            
            elif parameters[i,0][0] == 'N':
                with open('fittedparameters/restraints_'+parameters[i,0][1:]+'_methyl_charged.out') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'Fin':
                        charges = fitoutput[j+1:len(fitoutput)-6]
                        for k in range(0, len(charges)):
                            charges[k] = charges[k][:-2]
                        alpha  = np.genfromtxt('fittedparameters/alpha_'+parameters[i,0][1:]+'_methyl_charged.out.txt')
                        alpha = alpha[:len(alpha)-6]
                        if parameters[i,0][1:] == 'CYX':
                            charges = np.delete(charges, [16, 17, 18, 19, 20])
                            alpha = np.delete(alpha, [16, 17, 18, 19, 20])
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,4] = 0.0
                            parameters[i+k,5] = 0.0
                            parameters[i+k,6] = alpha[k]
                            parameters[i+k,7] = 0.0
                            parameters[i+k,8] = alpha[k]
                
            elif parameters[i,0][0] == 'n':
                with open('fittedparameters/restraints_'+parameters[i,0][1:]+'_methyl_neutral.out') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'Fin':
                        charges = fitoutput[j+1:len(fitoutput)-6]
                        for k in range(0, len(charges)):
                            charges[k] = charges[k][:-2]
                        alpha  = np.genfromtxt('fittedparameters/alpha_'+parameters[i,0][1:]+'_methyl_neutral.out.txt')
                        alpha = alpha[:len(alpha)-6]
                        if parameters[i,0][1:] == 'CYX':
                            charges = np.delete(charges, [16, 17, 18, 19, 20])
                            alpha = np.delete(alpha, [16, 17, 18, 19, 20])
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,4] = 0.0
                            parameters[i+k,5] = 0.0
                            parameters[i+k,6] = alpha[k]
                            parameters[i+k,7] = 0.0
                            parameters[i+k,8] = alpha[k]
            
            elif parameters[i,0][0] == 'A' and len(parameters[i,0][1:]) == 3:
                with open('fittedparameters/'+parameters[i,0][1:]+'ACE_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2:len(fitoutput)-6]
                        for k in range(0, len(charges)):
                            charges[k] = charges[k][:-2]
                        alpha  = np.genfromtxt('fittedparameters/alpha_'+parameters[i,0][1:]+'.out.txt')
                        alpha = alpha[0:len(alpha)-6]
                        if parameters[i,0][1:] == 'CYX':
                            charges = np.delete(charges, [16, 17, 18, 19, 20])
                            alpha = np.delete(alpha, [16, 17, 18, 19, 20])
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,4] = 0.0
                            parameters[i+k,5] = 0.0
                            parameters[i+k,6] = alpha[k]
                            parameters[i+k,7] = 0.0
                            parameters[i+k,8] = alpha[k]
            
            elif parameters[i,0][0] == 'B':
                with open('fittedparameters/'+parameters[i,0][1:]+'NME_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2+6:len(fitoutput)]
                        for k in range(0, len(charges)):
                            charges[k] = charges[k][:-2]
                        alpha  = np.genfromtxt('fittedparameters/alpha_'+parameters[i,0][1:]+'.out.txt')
                        alpha = alpha[6:len(alpha)]
                        if parameters[i,0][1:] == 'CYX':
                            charges = np.delete(charges, [10, 11, 12, 13, 14])
                            alpha = np.delete(alpha, [10, 11, 12, 13, 14])
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,4] = 0.0
                            parameters[i+k,5] = 0.0
                            parameters[i+k,6] = alpha[k]
                            parameters[i+k,7] = 0.0
                            parameters[i+k,8] = alpha[k]
            
            elif len(parameters[i,0]) == 3:
                with open('fittedparameters/restraints_'+parameters[i,0]+'.out') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'Fin':
                        charges = fitoutput[j+1+6:len(fitoutput)-6]
                        for k in range(0, len(charges)):
                            charges[k] = charges[k][:-2]
                        alpha  = np.genfromtxt('fittedparameters/alpha_'+parameters[i,0][:]+'.out.txt')
                        alpha = alpha[6:len(alpha)-6]
                        if parameters[i,0][:] == 'CYX':
                            charges = np.delete(charges, [10, 11, 12, 13, 14])
                            alpha = np.delete(alpha, [10, 11, 12, 13, 14])
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,4] = 0.0
                            parameters[i+k,5] = 0.0
                            parameters[i+k,6] = alpha[k]
                            parameters[i+k,7] = 0.0
                            parameters[i+k,8] = alpha[k]
                    
        current = parameters[i,0]
    
    FFpar = open('FFparameterswithduplicates.csv','w')
    for i in range(0, len(parameters)):
        FFpar.write(parameters[i,0])
        FFpar.write(',')
        FFpar.write(parameters[i,1])
        FFpar.write(',')
        FFpar.write(parameters[i,2])
        FFpar.write(',')
        FFpar.write(parameters[i,3])
        FFpar.write(',')
        FFpar.write(parameters[i,4])
        FFpar.write(',')
        FFpar.write(parameters[i,5])
        FFpar.write(',')
        FFpar.write(parameters[i,6])
        FFpar.write(',')
        FFpar.write(parameters[i,7])
        FFpar.write(',')
        FFpar.write(parameters[i,8])
        
            
        FFpar.write('\n')
    FFpar.close()


def removeduplicates():
    par = np.genfromtxt('FFparameterswithduplicates.csv', delimiter=',',dtype='U256')
    
    FFdict = {}
    for i in range(0, len(par)):
        if par[i,0]+par[i,1] not in FFdict:
            FFdict[par[i,0]+par[i,1]] = par[i,:]
    
    FFpar = open('FFparameters.csv','w')
    for key in FFdict:
        parameters = FFdict[key]
        FFpar.write(parameters[0])
        FFpar.write(',')
        FFpar.write(parameters[1])
        FFpar.write(',')
        FFpar.write(parameters[2])
        FFpar.write(',')
        FFpar.write(parameters[3])
        FFpar.write(',')
        FFpar.write(parameters[4])
        FFpar.write(',')
        FFpar.write(parameters[5])
        FFpar.write(',')
        FFpar.write(parameters[6])
        FFpar.write(',')
        FFpar.write(parameters[7])
        FFpar.write(',')
        FFpar.write(parameters[8])
        FFpar.write('\n')
    FFpar.close()

makeparameterfiler()
removeduplicates()