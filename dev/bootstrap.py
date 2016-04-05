# -*- coding: utf-8 -*-
"""
para sacar las barras de error a los parametros de calibracion.
se sabe que cuando se proyecta de VCA a Mapa hay un error las 
coordenadas x,y que tiee dist gaussiana que en pixeles tiene desv 
estandar s_N-1 = 3.05250940223 la idea es generara aleatoriamente 42 
puntos para entrenar ubicado en VCA, los transformamos segun los 
parametros optimos y se le suma el error. asi se obtienen los puntos 
Mapa para entrenar. En cada entrenamiento se guardan los parametros para despues hacer 
astadistica

@author: sebalander

"""
#%% IMPORTS
import numpy as np
from lmfit import minimize, Parameters
import pickle
from copy import deepcopy as copiar
import matplotlib.pyplot as plt
import sys

sys.path.append('../modules') # where modules are
from VCA2Earth_RelErr import VCA2Earth_RelErr
from forwardVCA2Map import forwardVCA2Map

if __name__ == '__main__':
    #%% LOAD DATA
    # load mu and covariance matrix
    data=np.loadtxt('comparoSalidas.txt').T
    #vamos a ver la correlacion entre error en x e y
    erx=data[4]-data[0]
    ery=data[5]-data[1]
    # para calcular la gaussiana eliptica
    X=np.array([erx,ery]) # lista de valores
    mu=np.array(np.matrix(np.mean(X,1)))[0] # valor medio
    SS=np.dot(X,X.T)-np.dot(mu,mu.T) #matriz de covarianza


    # load optimized parameters (best fit)
    prm=Parameters()
    with open('parametros.pkl', 'rb') as f:
	    prm = pickle.load(f)
    # aseguramos de variar los seis parametros a la vez
    prm['Px'].vary = prm['Py'].vary = prm['Pz'].vary=True
    prm['gamma'].vary = prm['beta'].vary = prm['alfa'].vary=True

    #%% PARAMETERS
    centro=1920/2 # posicion del centro de la imagen
    Rmax=873 # distancia maxima que pueden tener del centro de la imagen
    s=3.05250940223 # desviacion estandar de la gausiana de ruido
    N=10000# se va a hacer N veces en 'experimento'
    M=42 # cantidad de pares de puntos a usar
    XX=np.empty((4,M))
    PAR=np.empty((N,6)) # donde guardar los valores de los parametros

    print(N,'experimentos con',M,'puntos.')

    #%% LOOP
    for i in range(N):
	    print( '	experimento',i,'esimo')
	    t = np.random.random_sample(M)# variable auxiliar de donde sacar los radios
	    r = Rmax*np.sqrt(t)# radios generados
	    fi = 2*np.pi*np.random.random_sample(M)# angulos generados
	
	    # no tomar en cuenta la discretizacion en pixeles
	    # esto bien podria salir de buscar puntos de interes
	    XX[0] = r*np.cos(fi)+centro # posiciones en x
	    XX[1] = r*np.sin(fi)+centro # posisiones en y
	    # tengo los puntos en la imagen VCA, ahora a transformarlos y
	    # y ponerles ruido
	
	    XX[2],XX[3] = forwardVCA2Map(prm,XX[0],XX[1]) # con parametros optimos
	
    #	ruido=np.random.normal(0.0,s,(2,M))
	    #ruido de la gausiana bivaluada
	    ruido = np.random.multivariate_normal(mu,SS,M)
	    XX[2:] = XX[2:]+ruido.T
	
	    #ahora tengo los puntos listos para entrenar
	    prm_aux = copiar(prm) # hago una copia para modificar parametros
	
	    res = minimize(VCA2Earth_RelErr,prm_aux,args=([XX]))# optimizacion
	    # guardo parametros
	    PAR[i,:] = np.array([
                            	prm_aux['Px'].value,
                            	prm_aux['Py'].value,
                            	prm_aux['Pz'].value,
                            	prm_aux['gamma'].value,
                            	prm_aux['beta'].value,
                            	prm_aux['alfa'].value ])
	

    #%% PRINT PARAMETERS
    print( 'promedios y desviaciones estandar en este orden:')
    print( 'px,py,pz,gamma, beta, alfa')
    for i in range(6):
	    print( ' =',np.mean(PAR[:,i]),'\pm',np.std(PAR[:,i],ddof=1),'\\')

    """
    se obtiene
    promedios y desviaciones estandar en este orden:
    px,py,pz,gamma, beta, alfa
     = 297.404047304 \pm 4.43071704293 \
     = 272.780351563 \pm 7.35354637011 \
     = 29.7903688186 \pm 1.57777006693 \
     = 3.17515588912 \pm 0.0110500500912 \
     = -0.0237482769487 \pm 0.0131952778028 \
     = 1.68907876337 \pm 0.0464673979487 \
    """

    #%% PRINT PLOT
    fig, axis = plt.subplots(nrows=2,ncols=3);
    binposition=50
    for i in range(3):
	    axis[0,i].hist(PAR[:,i],binposition,normed=True,histtype='step')

    binsangle=50
    for i in range(3):
	    axis[1,i].hist(PAR[:,i+3],binposition,normed=True,histtype='step')

    plt.savefig('boots.eps')



