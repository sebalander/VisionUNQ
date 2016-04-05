# -*- coding: utf-8 -*-
"""
funciones_correccion.py declaraciones de funciones para el main loop_correccion

@author: Damián E. Stanganelli
"""
import sys

sys.path.append('../modules')  # where modules are
from map_sphrot import map_sphrot
from findMR import findMR
from map_sph import map_sph
from map_ptzv import map_ptzv




def warp(src, u, v, li, mi, si, ni, wi, fovi):
    '''
    Funcion que corrige una imagen fisheye centrandola en (u, v), usando
    el MUI
    '''
    # Parametros del Modelo Unificado de Imagenes
    l, m = li, mi
    u0, v0 = si/2, si/2
    # Dimension de la proyeccion esferica deseada (nxn)
    n = ni
    # Se mapean los puntos desde la esfera unitaria hacia el plano
    # imagen VCA
    U, V = map_sph(n, l, m, u0, v0)
    # Se proyectan los puntos desde el plano imagen VCA hacia la esfera
    # unitaria
    sph = cv2.remap(src, U, V, cv2.INTER_LINEAR)
    # Se calcula la matriz de rotacion para apuntar la esfera a un punto
    # (u, v) dado
    MR = findMR(u, v, l, m, u0, v0)
    # Se mapean los puntos desde la esfera rotada hacia la esfera
    # original
    Theta, Phi = map_sphrot(n, MR)
    # Se proyectan los puntos desde la esfera original hacia la rotada
    sphrot = cv2.remap(sph, Theta, Phi, cv2.INTER_LINEAR)
    # Dimension de la imagen en perspectiva deseada
    w = wi
    # Se mapean los puntos desde el plano imagen PTZ virtual hacia la
    # esfera rotada
    ThetaR, PhiR = map_ptzv(w, n, fovi)
    # Se proyectan los puntos desde la esfera rotada hacia el plano
    # imagen PTZv
    ptzv = cv2.remap(sphrot, ThetaR, PhiR, cv2.INTER_LINEAR)
    return ptzv


def MIUtheta(r, l, m):
    '''
    Funcion inversa que calcula theta en funcion de r, usando el Modelo
    Unificado
    theta(r),  Modelo Unificado de Imagenes
    Entradas:
    r: Distancia entre un punto en el plano imagen y el punto principal
    l, m: Parametros del Modelo Unificado de Imagenes
    Salidas:
    theta: angulo medido desde el sur de la esfera unitaria hasta el
    punto de interes
    '''
    r2 = r**2
    lm2 = (l+m)**2
    theta = np.arccos(((l+m)*np.sqrt(r2*(1-l**2)+lm2)-l*r2)/(r2+lm2))
    return theta

def MIUr(theta, l, m):
    '''
    Funcion que calcula r en funcion de theta,  usando el Modelo
    Unificado
    r(theta),  Modelo Unificado de Imagenes"
    Entradas:
    theta: angulo medido desde el sur de la esfera unitaria hasta el
    punto de interes
    l, m: Parametros del Modelo Unificado de Imagenes
    Salidas:
    r: Distancia entre un punto en el plano imagen y el punto
    principal
    '''
    r = (l+m)*np.sin(theta)/(l+np.cos(theta))
    return r

def map_sphrot(n, MR):
    '''
    Proyeccion de matriz de puntos en la esfera unitaria rotada hacia la
    esfera original
    Entradas:
    n: Dimension de la matriz de puntos deseada (nxn)
    MR: Matriz de rotacion
    Salidas:
    Thetar, Phir: Matriz de puntos de la esfera rotada proyectados en la
    esfera original
    '''
    # Rango del angulo theta: [0;pi/2) rad
    ThetaR_range = np.linspace(0, np.pi/2, n)
    # Rango del angulo phi: [-pi;pi) rad
    PhiR_range = np.linspace(-np.pi, np.pi, n)
    # Creacion de matriz de puntos en la esfera rotada usando los rangos
    # definidos
    ThetaR,  PhiR = np.meshgrid(ThetaR_range,  PhiR_range)
    # Coordenadas cartesianas de los puntos de la matriz
    P = np.array([np.sin(ThetaR)*np.cos(PhiR),
                  np.sin(ThetaR)*np.sin(PhiR),
                  -np.cos(ThetaR)])
    # Se aplica la rotacion a todos estos puntos
    PR = np.sum(MR.T[:, :, np.newaxis, np.newaxis] *
                P[:, np.newaxis, :, :], 0)
    # Calculos auxiliares para hallar las coordenadas esfericas de PR
    X = np.clip(np.sqrt(PR[0]**2 + PR[1]**2), 0, 1)
    Theta = np.where(PR[2] < 0, np.arcsin(X), 0)
    Phi = np.arctan2(PR[1], PR[0])
    # Se escalan a la imagen
    Theta = Theta/(np.pi/2)*n
    Phi = (Phi+np.pi)/(2*np.pi)*n
    # Se convierte en float32 para la funcion remap
    Theta, Phi = np.float32((Theta, Phi))
    return Theta, Phi


def map_sph(n, l, m, u0, v0):
    '''
    Proyeccion de matriz de puntos en la esfera hacia el plano de la
    imagen VCA,  MUI"
    Entradas:
    n: Dimension de la matriz de puntos deseada (nxn)
    l, m: Parametros del Modelo Unificado de Imagenes
    u0, v0: Punto principal de la imagen VCA
    Salidas:
    U, V: Matriz de puntos de la esfera unitaria proyectados en el
                                                     plano imagen VCA
    '''
    # Rango del angulo theta: [0;pi/2) rad
    Theta_range = np.linspace(0, np.pi/2, n)
    # Rango del angulo phi: [-pi;pi) rad
    Phi_range = np.linspace(-np.pi, np.pi, n)
    # Creacion de matriz de puntos en la esfera usando los rangos
    # definidos
    Theta, Phi = np.meshgrid(Theta_range, Phi_range)
    # Proyeccion de los puntos hacia la imagen VCA
    R = MIUr(Theta, l, m)
    U, V = R*np.cos(Phi)+u0,  R*np.sin(Phi)+v0
    # Se convierte en float32 para la funcion remap
    U, V = np.float32((U, V))
    return U, V


def map_ptzv(w, n, fov=30):
    '''
    Proyeccion de matriz de puntos en el plano imagen PTZ virtual hacia
    la esfera,  MUI
    Entradas:
    w: Dimension de la imagen en perspectiva deseada (w, w)
    n: Dimension de la matriz de puntos de la proyeccion esferica
                                                                (nxn)
    fov: Campo de vision a simular, las camaras PTZ tradicional
                                                        tienen 70 deg
    Salidas:
    Theta, Phi: Matriz de puntos del plano imagen PTZ proyectados en
    la esfera unitaria
    '''
    # Calculo de los parametros del MUI
    l, m = 0, (w/2)/np.tan(fov*np.pi/180)
    # Calculo del punto principal de la imagen PTZ virtual
    u0 = v0 = w/2
    # Creacion de matriz de puntos en la imagen PTZ virtual
    U, V = np.meshgrid(range(w), range(w))
    # Calculo de las coordenada polares para cada punto
    R = np.sqrt((U-u0)**2+(V-v0)**2)
    Phi = np.arctan2(V-v0, u0-U)
    # Proyeccion de los puntos hacia la esfera
    Theta = MIUtheta(R, l, m)
    # Se escalan a la imagen
    Theta = Theta/(np.pi/2)*n
    Phi = (np.pi-Phi)/(2*np.pi)*n
    # Se convierte en float32 para la funcion remap
    Theta, Phi = np.float32((Theta, Phi))
    return Theta, Phi


def calcularGlobales(li, mi, si, ni, wi, fovi):
    """
    Funcion que calcula las variables globales comunes a todos los mapeos
    """
    global l, m, u0, v0, n, U, V, w, ThetaR, PhiR
    # Parametros del Modelo Unificado de Imagenes
    l, m = li, mi
    u0, v0 = si/2, si/2
    # Dimension de la proyeccion esferica deseada (nxn)
    n = ni
    # Se mapean los puntos desde la esfera unitaria hacia el plano
    # imagen VCA
    U, V = map_sph(n, l, m, u0, v0)
    # Dimension de la imagen en perspectiva deseada
    w = wi
    # Se mapean los puntos desde el plano imagen PTZ virtual hacia la
    # esfera rotada
    ThetaR, PhiR = map_ptzv(w, n, fovi)


def corregir(src, u, v):
    '''
    Funcion que corrige una imagen fisheye centrandola en (u, v),
    usando el MUI
    '''
    global l, m, u0, v0, n, U, V, w, ThetaR, PhiR
    # Se proyectan los puntos desde el plano imagen VCA hacia la esfera
    # unitaria
    sph = cv2.remap(src, U, V, cv2.INTER_LINEAR)
    # Se calcula la matriz de rotacion para apuntar la esfera a un punto
    # (u, v) dado
    MR = findMR(u, v, l, m, u0, v0)
    # Se mapean los puntos desde la esfera rotada hacia la esfera
    # original
    Theta, Phi = map_sphrot(n, MR)
    # Se proyectan los puntos desde la esfera original hacia la rotada
    sphrot = cv2.remap(sph, Theta, Phi, cv2.INTER_LINEAR)
    # Se proyectan los puntos desde la esfera rotada hacia el plano
    # imagen PTZv
    ptzv = cv2.remap(sphrot, ThetaR, PhiR, cv2.INTER_LINEAR)
    return ptzv


def findMR(u, v, l, m, u0, v0):
    '''
    Matriz de rotacion"
    Entradas:
    u, v: Punto de interes en la imagen
    l, m: Parametros del Modelo Unificado de Imagenes
    u0, v0: Punto principal de la imagen VCA
    Salidas:
    MR: Matriz de rotacion
    '''
    # Coordenadas esfericas del punto de interes proyectado en la esfera
    # unitaria
    u, v = u0*2-u, v0*2-v
    phi = np.arctan2(v-v0, u-u0)
    r = np.sqrt((u-u0)**2+(v-v0)**2)
    theta = MIUtheta(r, l, m)
    # Coordenadas cartesianas del versor proyectado
    P = np.array([np.sin(theta)*np.cos(phi),
                  np.sin(theta)*np.sin(phi),
                  -np.cos(theta)])
    # Versor que apunta al sur de la esfera
    Pz = np.array([0, 0, -1])
    # Versor de rotacion
    Pk = np.cross(P, Pz)
    Pk = Pk if np.sum(Pk) == 0 else Pk/np.sqrt(np.sum(Pk**2))
    # Calculo de la matriz de rotacion
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1-c
    kx, ky, kz = Pk
    MR = np.array([[kx*kx*v+c, kx*ky*v-kz*s, kx*kz*v+ky*s],
                   [kx*ky*v+kz*s, ky*ky*v+c,  ky*kz*v-kx*s],
                   [kx*kz*v-ky*s, ky*kz*v+kx*s, kz*kz*v+c]])
    # Como la camara apunta hacia abajo, se debe hacer la siguiente
    # correccion
    if np.sum(Pk) != 0:
        beta = phi+np.pi/2
        cb = np.cos(beta)
        sb = np.sin(beta)
        Rz = np.array([[cb, -sb, 0], [sb, cb, 0], [0, 0, 1]])
        MR = np.dot(MR, Rz)
    return MR

