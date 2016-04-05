# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:03:16 2015

Plotea no se que

@author: jew
"""

# Importo Pickle
import pickle
import cv2

# Numpy
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import binned_statistic

def dumpVar(name, var):
    filehandler = open("vars/" + name + ".obj","wb")
    pickle.dump(var,filehandler)
    filehandler.close()

def importVar(name, var):
    try:
        file = open("vars/" + name + ".obj",'rb')
        object_file = pickle.load(file)
        file.close()
    except:
        object_file = var
    return object_file

try:
    dumpVar('_res', res[1][1])
    dumpVar('_vel2', vg.pv.kp_dist_vect)
    dumpVar('_kphist', vg.pv.kp_hist)
    dumpVar('_ocu2', vg.pv.bf_vect)
    dumpVar('_imhi', vg.im_hist)
    dumpVar('_imbf', vg.im_bf)
    dumpVar('_imbf2', vg.im_bf2)
    dumpVar('_imframe', vg.im_frame)
    dumpVar('_immask', vg.im_mask)
    dumpVar('_tra', vg.im_transf)
    dumpVar('_lab', vg.im_label)
    dumpVar('_neti', vg.net_input)
    dumpVar('_labels', vg.labels[0])
except:
    ocu = importVar('_ocu','_ocu')
    vel = importVar('_vel','_vel')
    ocu2 = importVar('_ocu2','_ocu2')
    vel2 = importVar('_vel2','_vel2')
    tra = importVar('_tra','_tra')
    imhi = importVar('_imhi','_imhi')
    imbf = importVar('_imbf','_imbf')
    imbf2 = importVar('_imbf2','_imbf2')
    imframe = importVar('_imframe','_imframe')
    immask = importVar('_immask','_immask')
    resa = importVar('_res','_res')
    lab = importVar('_lab','_lab')
    neti = importVar('_neti','_neti')
    labels = importVar('_labels','_labels')
    kphist = importVar('_kphist','_kphist')
    
    ocu = imframe.copy()
    ocu[imbf2>0,2] = 255
    ocu = cv2.cvtColor(ocu, cv2.COLOR_BGR2RGB)
    
    tot = imframe.copy()
    tot[immask>0,2] = 255
    tot = cv2.cvtColor(tot, cv2.COLOR_BGR2RGB)
        
#    plt.figure()
#    plt.imshow(cv2.cvtColor(resa[a1:a2,a3:a4], cv2.COLOR_BGR2RGB))
#    plt.figure()
#    plt.imshow(cv2.cvtColor(tra, cv2.COLOR_BGR2RGB))
#    plt.figure()
#    plt.imshow(imhi[a1:a2,a3:a4])
#    plt.figure()
#    plt.imshow(imbf[a1:a2,a3:a4])
#    plt.figure()
#    plt.imshow(imbf2[a1:a2,a3:a4]>0)
    plt.figure()
    plt.imshow(ocu)
    plt.figure()
    plt.imshow(tot)
    
#    means = [0,0,0,0]
#    estan = [0,0,0,0]
#    ind = np.arange(4)
#    width = 0.25
    
#    plt.figure()
#    vel2 = np.vstack(velf[4000:10100])
#    ocu2 = np.vstack(ocu[4000:10100])
#    plt.plot(np.array(vel2)/110)
#    plt.plot(np.array(ocu2)/15525)
#    plt.xlim(0,len(vel2))
#    plt.ylim(0,1.2)
#    plt.legend(['Velocidad promedio normalizada','Llenado normalizado de la via'])
