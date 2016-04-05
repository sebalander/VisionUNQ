# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
se grafica histograma del error en x e y de entrenar con 42 puntos. 
tambien se reporta el valor medio y desviación estándar sin bias.

@author: sebalander
'''
#%% IMPORTS
import numpy as np
from lmfit import Parameters
import pickle
import matplotlib.pyplot as plt

#%% DECLARATION
def prob(x,y):
    '''
    Funcion que devuelve la probabilidad en la coordenada (x,y).
    toma matriz de covarianza invertida a ZZ, al denominador de
    la expresion denom=2*pi*sqrt(det(mat.covarianza)), y el centro
    de la distribucion mu (bidimensional)
    '''
    x2=x-mu[0] #centrar coordenadas
    y2=y-mu[1]
#	print x2,y2
    poli=ZZ[0,0]*x2**2 + 2*ZZ[0,1]*x2*y2 + ZZ[1,1]*y2**2 # X.T·ZZ·X
#	print poli
    p=np.exp(-0.5*poli)/denom
#	print p
    return p


if __name__ == '__main__':
    #%% LOAD DATA
    # se cargan 6 columnas, de apres x-y, en este orden; salidas deseadas, obtenidas antes de optimizar (cond iniciales) y despues de optimizar.
    data=np.loadtxt('../resources/comparoSalidas.txt').T

    #%% Vamos a ver la correlacion entre error en x e y
    erx=data[4]-data[0]
    ery=data[5]-data[1]

    # para calcular la gaussiana eliptica
    X=np.array([erx,ery]) # lista de valores
    mu=np.array(np.matrix(np.mean(X,1)))[0] # valor medio
    SS=np.dot(X,X.T)-np.dot(mu,mu.T) #matriz de covarianza
    ## SS= matrix([[ 180.27681784,  221.15050451],
    #              [ 221.15050451,  594.68508288]])
    [l1,l2] , [eig1,eig2] = np.linalg.eig(SS) #extract autovalores y autovectores
    ZZ=np.linalg.inv(SS) # tengo la inversa para mas comodidad
    #el denominador en la expresion de la gaussiana multivaluada
    denom=2*np.pi*np.sqrt(np.linalg.det(SS))
    #%% PLOT
    # crear ejes para las graficas y eso.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # grafico para ver dependencia
    ax.scatter(erx,ery)
    # ahora las curvas de nivel
    delta = 0.5
    x = np.arange(-10, 10, delta)
    y = np.arange(-20, 15, delta)
    xx, yy = np.meshgrid(x, y)
    zz = prob(xx,yy)
    #grafico curvas
    plt.contour(xx,yy,zz)

    plt.savefig('../resources/erxycorrel.eps')

    #%% 
    # se cargan valores optimos de parametros, aca esta la posicion de la
    # camara
    prm=Parameters()
    with open('../resources/parametros.pkl', 'rb') as f:
	    prm = pickle.load(f)

    xy=np.insert(erx,0,ery)
    N=len(xy) # es 84, la cantidad de datos

    print('valor medio del error y desviacion estandar (con y sin bias)')
    s=np.std(xy,ddof=1) # desviacion estandas unbiased
    print(np.mean(xy),np.std(xy) , s)


    bins=50
    Inter=20
    Xgaus=np.linspace(-Inter,Inter,bins)
    Ygaus=np.exp(-(Xgaus/s)**2)/np.sqrt(2*np.pi)/s * bins/Inter*0.5

    #del ax.lines[0:1] # borro este eje con el contenido anterior # no anduvo con scatter
    fig.clear() # esto si anda para limpiar la figura

    ax = fig.add_subplot(111) # agrego nuevo eje
    ax.hist(xy,Xgaus,normed=True,histtype='step')
    ax.plot(Xgaus,Ygaus)
    plt.savefig('../resources/errorhist.eps')


