import numpy as np

def makeparameterfiler():
    parameters = np.genfromtxt('FFparameterstemplate.csv',delimiter=';',dtype=str)
    current = 'None'
    for i in range(1,len(parameters)):
        
        if current != parameters[i,0]:
            if parameters[i,0][0] == 'C' and parameters[i,0][1] != 'Y':
                with open('fittedparameters/'+parameters[i,0][1:]+'chargedmethyl_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2+6:len(fitoutput)]
                        alpha  = np.genfromtxt('fittedparameters/alpha_'+parameters[i,0][1:]+'_charged_methyl.out.txt')[6:]
                        if parameters[i,0][1:] == 'CYX':
                            charges = np.delete(charges, [10, 11, 12, 13, 14])
                            alpha = np.delete(alpha, [10, 11, 12, 13, 14])
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,4] = 0.0
                            parameters[i+k,5] = 0.0
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,5] = 0.0
                            parameters[i+k,3] = alpha[k]
                                
            elif parameters[i,0][0] == 'c':
                with open('fittedparameters/'+parameters[i,0][1:]+'neutralmethyl_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2+6:len(fitoutput)]
                        alpha  = np.genfromtxt('fittedparameters/alpha_'+parameters[i,0][1:]+'_neutral_methyl.out.txt')[6:]
                        if parameters[i,0][1:] == 'CYX':
                            charges = np.delete(charges, [10, 11, 12, 13, 14])
                            alpha = np.delete(alpha, [10, 11, 12, 13, 14])
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,4] = 0.0
                            parameters[i+k,5] = 0.0
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,5] = 0.0
                            parameters[i+k,3] = alpha[k]
                            
            elif parameters[i,0][0] == 'N':
                with open('fittedparameters/'+parameters[i,0][1:]+'methylcharged_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2:len(fitoutput)-6]
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
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,5] = 0.0
                            parameters[i+k,3] = alpha[k]
                
            elif parameters[i,0][0] == 'n':
                with open('fittedparameters/'+parameters[i,0][1:]+'methylneutral_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2:len(fitoutput)-6]
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
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,5] = 0.0
                            parameters[i+k,3] = alpha[k]
                
            else:
                with open('fittedparameters/'+parameters[i,0]+'_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2+6:len(fitoutput)-6]
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
                            parameters[i+k,3] = alpha[k]
                            parameters[i+k,5] = 0.0
                            parameters[i+k,3] = alpha[k]
                    
        current = parameters[i,0]
    
    FFpar = open('FFparameters.csv','w')
    for i in range(0,len(parameters)):
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

makeparameterfiler()