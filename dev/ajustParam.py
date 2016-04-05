#!/usr/bin/env python
'''
Created on Thu Feb 11 2016

Fits parameters of the transformation function using calibration 
points.

@author:sebalander
'''

#%% IMPORTS
import numpy as np
from lmfit import minimize, Parameters
import pickle
import sys

sys.path.append('../modules') # where modules are
from VCA2Earth_RelErr import VCA2Earth_RelErr
from paramflat import paramflat


if __name__ == '__main__':
    #%% LOAD DATA

    # cargo todos los puntos para ajustar desde un archivo
    XX=np.loadtxt('puntosCalibracion.txt').T 
    # se cargan valores iniciales para parametros, a ojo
    prm=Parameters()
    with open('parametrosiniciales.pkl', 'rb') as f:
	    prm = pickle.load(f)

    print('valores iniciales de los parametros')
    for par in prm:
        print(par, prm[par].value)

    #%% MINIMIZE W/ANGLES
    res = minimize(VCA2Earth_RelErr, prm, args=([XX]) )#,maxfev=500000)
    print(res.success, res.message)

    print( '\n\nparametros habiendo ajustado solo los angulos')
    for par in prm: print(par, prm[par].value)

    #%% MINIMIZE W/POSITION
    prm['Px'].vary = prm['Py'].vary = prm['Pz'].vary=True
    prm['gamma'].vary = prm['beta'].vary = prm['alfa'].vary=False

    res=minimize(VCA2Earth_RelErr,prm,args=([XX]))#,maxfev=500000)
    print(res.success, res.message)

    print ('\n\nparametros habiendo ajustado solo las posiciones respecto a res anteriores')
    for par in prm: print( par, prm[par].value)

    #%% MINIMIZE W/POSITION AND ANGLES
    prm['gamma'].vary = prm['beta'].vary = prm['alfa'].vary=True

    res=minimize(VCA2Earth_RelErr,prm,args=([XX]))#,maxfev=500000)
    print( res.success, res.message)

    print ('\n\nparametros habiendo ajustado angulos y  posiciones respecto a res anteriores')
    for par in prm: print( par, prm[par].value)

    #%% SAVE PARAMETERS
    fil='parametrosOptimizados'
    with open(fil+'.pkl', 'wb') as f:
	    pickle.dump(prm, f)

    TT = paramflat(prm) # transform parameters to flat list
    with open(fil+'.txt','wb') as f:
      np.savetxt(f,TT) # save transform constants
