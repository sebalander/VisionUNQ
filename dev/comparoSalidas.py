#!/usr/bin/env python
'''
Created on Thu Feb 11 2016

Takes coordinates in VCA image and transforms them to map coordinates
after loading transformation parameters from file. Uses all 
calibration points to compare outputs with initial condition 
parameters and optimized paramters.

@author:sebalander
'''
#%% IMPORTS
import numpy as np
import sys
## uncomment next two lines if loading params from pkl file
# from lmfit import  Parameters 
# import pickle

sys.path.append('../modules') # where modules are
from VCA2Map import VCA2Map
# from forwardVCA2Map import forwardVCA2Map

if __name__ == '__main__':
    #%% LOAD PARAMETERS. Optimized and non optimal for comparison
    # se cargan valores optimizados 
    #prm=Parameters()
    #prm0=Parameters()

    ## Leer parametros como listas

    with open('../resources/parametrosiniciales.txt', 'rb') as f:
	    prm0 = np.loadtxt(f)
    with open('../resources/parametrosOptimizados.txt', 'rb') as f:
	    prm = np.loadtxt(f)

    #%% REORGANIZE DATA
    # to recover original data:
    const0 = prm0[:5]
    T0 = prm0[5:14].reshape(3,3)
    T_10 = prm0[14:].reshape(3,3)
    const = prm[:5]
    T = prm[5:14].reshape(3,3)
    T_1 = prm[14:].reshape(3,3)



    # print( forwardVCA2Map(prm,1360,970),
    #      "que deberia ser aprox", 300, 310, "(cercano)")
    # print( forwardVCA2Map(prm,415,300),
    #      "que deberia ser aprox?", 60, 25, "(lejano)")

    #print 'para otros puntos, se calcula el error relativo'

    # cargo todos los puntos para ajustar desde un archivo
    XX=np.loadtxt('../resources/puntos.txt').T

    # tomo las coordenadas que deberian dar
    XXdeseado=XX[2:4] 

    # calculo lo que da el mapeo con cond inic
    XXout=VCA2Map(const0,T0,XX[0],XX[1])

    # calculo lo que da el mapeo con cond
    XXout2=VCA2Map(const,T,XX[0],XX[1]) 

    np.savetxt('../resources/comparosalidas.txt',
               np.concatenate((XXdeseado,XXout,XXout2)).T)









