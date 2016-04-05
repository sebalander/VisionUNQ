#!/usr/bin/env python
'''
Created on Thu Feb 11 2016

Fits parameters of the transformation function using calibration points.
Uses less apropiate initial conditions for fitting.

@author:sebalander
'''
#%% IMPORTS
import numpy as np
import random as rnd
from lmfit import minimize, Parameters
import pickle, copy
import sys

sys.path.append('../modules') # where modules are
from VCA2Earth_RelErr import VCA2Earth_RelErr
from VCA2Earth_ERR import VCA2Earth_ERR

if __name__ == '__main__':
    #%% PARMETERS
    N=6 # cantidad de puntos con los que hacer la optimizacion
    M=20000 # cantidad de veces que hacer la optimizacion
    XXd=np.loadtxt('puntos.txt').T # cargo todos los puntos para ajustar desde un archivo
    XXl=list(XXd.T) # to get sample make list

    #print XX
    # load not so optimal initial values
    prm=Parameters()
    with open('parametroshard.pkl', 'rb') as f:
        prm = pickle.load(f)

    # se imprime
    print('valores iniciales de los parametros')
    for par in prm: print(par, prm[par].value)

    #%% SET TO VARY EVERYTHING
    prm['Px'].vary = prm['Py'].vary = prm['Pz'].vary=True
    prm['gamma'].vary = prm['beta'].vary = prm['alfa'].vary=True

    #parfin=np.empty([M,6])
    #donde guardar los resultados, al final siempre, para poder ir agregado
    guardparfin=open('AjusteParametrosHard.txt','w')
    #%% LOOP
    for i in range(M):
        # se eligen N puntos para 'entrenar' el algoritmo
        puntos = np.array(rnd.sample(XXl,N)).T
    #    print puntos
    #    print "\n\niteracion%d\n"%(i),puntos
        
        #copio para optimizar parametros auxiliares desde mismas cond iniciales
        prmaux=copy.deepcopy(prm)
        # pongo el ultimo angulo en random (cama siempre apunta hacia abajo)
        prmaux['alfa'].value=rnd.uniform(-np.pi,np.pi)
        
        # calculo el error antes de optimizar
        E_before=VCA2Earth_ERR(prmaux,puntos)
        #impirmo antesd e optimizar
        parini="%f %f %f %f %f %f %f %f\n"%(prmaux['gamma'].value,prmaux['beta'].value,prmaux['alfa'].value, prmaux['Px'].value,prmaux['Py'].value,prmaux['Pz'].value,E_before[0],E_before[1])
        print("parini %d es"%(i),parini)
        
        # se optimiza
        res=minimize(VCA2Earth_RelErr,prmaux,args=([XXd]))#,maxfev=500000)
        
        
        # calculo el error despues de optimizar
        E_after=VCA2Earth_ERR(prmaux,puntos)
        # string donde van los paametros despues de optimizar
        parfin="%f %f %f %f %f %f %f %f\n"%(prmaux['gamma'].value,prmaux['beta'].value,prmaux['alfa'].value, prmaux['Px'].value,prmaux['Py'].value,prmaux['Pz'].value,E_after[0],E_after[1])
        #muestro en pantalla
        print('parfin %d es'%(i), parfin)
        
        guardparfin.write(parfin) # se escribe al archivo

    #cierro archivo
    guardparfin.close()

