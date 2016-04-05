# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:26:23 2015

Plots data from blobs

@author: jew
"""

# Importo Pickle
import pickle
import cv2

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Numpy
import numpy as np

import matplotlib.pyplot as plt

def importVar(name, var):
    try:
        file = open("vars/" + name + ".obj",'rb')
        object_file = pickle.load(file)
        file.close()
    except:
        object_file = var
    return object_file


if __name__ == '__main__':
    data_centers = importVar('_dc','_dc')
    data_areas = importVar('_da','_da')
    data_size = importVar('_ds','_ds')
    data_peri = importVar('_dp','_dp')

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    salida = []

    for k in range(0,len(data_size)):
        xs = data_size[k][0]
        ys = data_size[k][1]
        zs = data_peri[k]**2
        cs = data_centers[k][1]
        ca = data_areas[k]
        try:
            salida.append(1.*ca/zs)
        except:
            pass
        #ax.scatter(xs, ys, zs)
        #plt.scatter(ca, zs, marker='.')
        
    salida = np.array(salida)
    plt.hist(salida[salida<100],500)

    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    #plt.show()
