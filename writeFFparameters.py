import numpy as np

def makeparameterfiler():
    # REMEMBER TO MAKE FIX FOR CCYX and cCYX
    
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
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            
            elif parameters[i,0][0] == 'c':
                with open('fittedparameters/'+parameters[i,0][1:]+'neutralmethyl_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2+6:len(fitoutput)]
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                            
            elif parameters[i,0][0] == 'N':
                with open('fittedparameters/'+parameters[i,0][1:]+'methylcharged_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2:len(fitoutput)-6]
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                
            elif parameters[i,0][0] == 'n':
                with open('fittedparameters/'+parameters[i,0][1:]+'methylcharged_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2:len(fitoutput)-6]
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                
            else:
                with open('fittedparameters/'+parameters[i,0]+'_q_out.txt') as f:
                    fitoutput = list(f)
                for j in range(0,len(fitoutput)):
                    if fitoutput[j][0:3] == 'nit':
                        charges = fitoutput[j+2+6:len(fitoutput)-6]
                        for k in range(0,len(charges)):
                            parameters[i+k,2] = charges[k]
                    
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