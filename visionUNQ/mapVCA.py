# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 2016

Declarations of functions that transform between VCA and map
coordinates.

@author: sebalander
"""
#%% IMPORTS
import numpy as np

#%% DECLARATION
def paramflat(parametros):
    '''
    funcion que retorna los parametros necesarios para hacer las
    transformacions, centro de la imagen, constante k, posicion de
    la camara y las matrices de transformacion; para acceder sin
    tener que hacer calculos posteriores de nuevo cada vez.
    '''
    # get angles
    alfa=parametros['alfa'].value
    beta=parametros['beta'].value
    gamma=parametros['gamma'].value

    # calculate trig functions
    ca=np.cos(alfa)
    cb=np.cos(beta)
    cg=np.cos(gamma)
    sa=np.sin(alfa)
    sb=np.sin(beta)
    sg=np.sin(gamma)

    # he aqui la dichosa matriz de transformacion forward
    T = np.matrix( [ [ca*cb,ca*sb*sg-sa*cg,ca*sb+sa*sg],
    [sa*cb,sa*sb*sg+ca*cg,sa*sb*cg-ca*sg],
    [-sb,cb*sg,cb*cg]])
    T_1 = np.linalg.inv(T)
    # save in parametros class the transforma matrices
    #  parametros["T"].value = T
    #  parametros["T_1"].value = np.linalg.inv(T)[:,0:2]
    # only transform parameters for x,y are relevant for the inverse 

    # to save also other relevan parameters
    cen = parametros['cen'].value
    k = parametros['k'].value
    Px = parametros['Px'].value
    Py = parametros['Py'].value
    Pz = parametros['Pz'].value
    #return matrices as flat list
    L0=np.array(T.flatten())[0,:]
    L1=np.array(T_1.flatten())[0,:]
    TT = np.concatenate(([cen,k,Px,Py,Pz],L0,L1))
    # to recover original data:
    # cen,k,Px,Py,Pz = TT[:5]
    # T = TT[5:14].reshape(3,3)
    # T_1 = TT[14:].reshape(3,3)
    return TT

#%% DECLARATION
def VCA2Map(xI,yI,prm):
    """
    Funtion transforms coordinates in VCA image to map, less 
    cumbersome than ForwardVCA2Map. Parameters are provided as 
    arguments.
    
    Parameters
    ----------
    prm : vector consisting of center, k, Px, Py, Pz, and matrices T and T_1
            flattened.
        If the image to be processed has been resized (original is 1920), 
        must rescale center and k (possibly Px, Py, Pz too?)
    xI,yI : coordinates to be transformed.   
    
    Returns
    -------
    xM,yM : coordinates in map frame of reference.
    
    See Also
    --------
    
    Notes
    -----
    
    Examples
    --------
    import numpy as np
    # Read parameters from file
    
    with open('../resources/parametrosOptimizados.txt', 'rb') as f:
        prm = np.loadtxt(f)
    
    # apply transformation to points xI, yI
    [xM, yM] = VCA2map(xI, yI) 
    """
    
    # Auxililar calculations on panoramic coordinates
    xcen,ycen = xI-prm[0],yI-prm[0]
    r = np.sqrt(xcen**2+ycen**2)
    fi = np.arctan2(ycen,xcen)
    
    ## ahora a pasar a esfericas en el marco de referencia de la camara
    t = 2*np.arctan(r/prm[1]) 
    Ct = np.cos(t) # coseno de theta
    St = np.sin(t) # seno de theta
    
    Cfi = np.cos(fi)
    Sfi = np.sin(fi)
    
    ## salteandome las cartesianas paso directamente al mapa
    T = prm[5:14].reshape(3,3) # matrix for VCA->Map rotation
    Rho = -prm[4]/(St*(T[2,0]*Cfi+T[2,1]*Sfi)+Ct*T[2,2])
    xM = Rho*(St*(T[0,0]*Cfi+T[0,1]*Sfi)+T[0,2]*Ct)+prm[2]
    yM = Rho*(St*(T[1,0]*Cfi+T[1,1]*Sfi)+T[1,2]*Ct)+prm[3]
    
    return xM,yM

#%% DECLARATION
def map2VCA(xM, yM, prm):
    """
    Funtion transforms coordinates in Map to VCA image. Parameters are
    provided as arguments.
    
    Parameters
    ----------
    prm : vector consisting of center, k, Px, Py, Pz, and matrices T and T_1
            flattened.
        If the image to be processed has been resized (original is 1920), 
        must rescale center and k (possibly Px, Py, Pz too?)
    xM, yM : coordinates to be transformed.   
    
    Returns
    -------
    xM, yM : coordinates in map frame of reference.
    
    See Also
    --------
    
    Notes
    -----
    
    Examples
    --------
    import numpy as np
    # Read parameters from file
    with open('parametrosoptimizados.txt', 'rb') as f:
        prm = np.loadtxt(f)
    
    # apply transformation to points xI, yI
    [xI, yI] = map2VCA(xM, yM) 
    """
    
    Xaux = np.matrix([xM,yM,0] - prm[2:5]).transpose()
    
    T_1 = prm[14:].reshape(3,3) # matrix for Map->VCA rotation
    
    XC = T_1.dot(Xaux) # camera coordinates
    
    fi = np.arctan2(XC[1],XC[0]) # angle in image
    t = np.arctan2(np.sqrt(XC[0]**2+XC[1]**2),XC[2])
    
    r = prm[1]*np.tan(t/2) # radius in image
    
    xVCA = prm[0] + r*np.cos(fi)
    yVCA = prm[0] + r*np.sin(fi)
    
    return xVCA[0,0],yVCA[0,0]

#%% DECLARE FUNCTION
def VCA2Earth_RelErr(prm,XX):
    '''
    Returns relative error weighted by distance.
    '''
    # Execute forward transformation
    XM_estimate = forwardVCA2Map(prm,XX[0],XX[1])
    
    # Distance between camera and desired transformed position
    d = ( np.sqrt(  (XX[2]-prm['Px'].value)**2
        + (XX[3]-prm['Py'].value)**2
        + prm['Pz'].value**2) )
    
    # Distance between obtained and desired position
    e = np.sqrt((XM_estimate[0]-XX[2])**2+(XM_estimate[1]-XX[3])**2)
    
    # Return error weighted by 1/distance
    return e/d


#%% DECLARATION
def VCA2Earth_ERR(prm,XX):
    '''
    Returns total erros for a list of points
    '''
    E=np.array([0.0,0.0]) # donde acumular errores totales, normal y relativo
    for xx in XX.T:
        E+=VCA2Earth_err(prm,xx) # acumulo para este punto solo

    return E #retorno el total


#%% DECLARATION
def VCA2Earth_err(prm,XX):
    '''
    Return error and relative error weighted by distance
    '''
    # Execute forward transformation
    XM_estimate = forwardVCA2Map(prm,XX[0],XX[1])
    
    # Distance between camera and desired transformed position
    d = np.sqrt( (XX[2]-prm['Px'].value)**2
                + (XX[3]-prm['Py'].value)**2
                + prm['Pz'].value**2) 
    
    # Distance between obtained and desired position
    er = np.sqrt( (XM_estimate[0]-XX[2])**2
                   + (XM_estimate[1]-XX[3])**2  )
    
    return np.array([er,er/d])


#%% DECLARATION
def forwardVCA2Map(parametros,xI,yI):
    """
    Transforms coordinates from VCA image to Map frame of reference.
    Use VCA2Map instead.
    
    forwardVCA2Map(parametros,xI,yI) -> xM,yM
    
    Parameters
    ----------
    parametros : pickle stricture with transformation paramters.
    xI,yI : Floats, coordinate pair to be transformed.
    
    Returns
    -------
    xM,yM : Floats, coordinate pair in map frame of reference.
    
    See Also
    --------
    
    Notes
    -----
    
    Examples
    --------
    """
    
    cen=parametros['cen'].value
    k=parametros['k'].value
    alfa=parametros['alfa'].value
    beta=parametros['beta'].value
    gamma=parametros['gamma'].value
    Px=parametros['Px'].value
    Py=parametros['Py'].value
    Pz=parametros['Pz'].value

    ##calculos auxiliares sobre la imagen panoramica
    xcen,ycen=xI-cen,yI-cen
    r=np.sqrt(xcen**2+ycen**2)
    fi=np.arctan2(ycen,xcen)

    ## ahora a pasar a esfericas en el marco de referencia de la camara
    t=2*np.arctan(r/k) 
    Ct=np.cos(t) # coseno de theta
    St=np.sin(t) # seno de theta

    # para ahorrar cuentas y notacion, saco senos y cosenos una sola vez
    ca=np.cos(alfa)
    cb=np.cos(beta)
    cg=np.cos(gamma)
    sa=np.sin(alfa)
    sb=np.sin(beta)
    sg=np.sin(gamma)

    # he aqui la dichosa matriz
    T11,T12,T13=ca*cb,ca*sb*sg-sa*cg,ca*sb+sa*sg
    T21,T22,T23=sa*cb,sa*sb*sg+ca*cg,sa*sb*cg-ca*sg
    T31,T32,T33=-sb,cb*sg,cb*cg

    ## salteandome las cartesianas paso directamente al mapa
    Rho = -Pz/(St*(T31*np.cos(fi)+T32*np.sin(fi))+Ct*T33)
    xM = Rho*(St*(T11*np.cos(fi)+T12*np.sin(fi))+T13*Ct)+Px
    yM = Rho*(St*(T21*np.cos(fi)+T22*np.sin(fi))+T23*Ct)+Py

    return xM,yM
